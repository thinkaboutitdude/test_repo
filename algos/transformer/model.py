import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_embd: int
    n_layer: int
    n_query_head: int
    attn_pdrop: float
    resid_pdrop: float
    block_size: int
    rope: bool


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class RotaryPositionalEmbeddings(nn.Module):
    """ 
    RoPE implementation as introduced in the paper RoFormer: Enhanced Transformer with Rotary Position Embedding.
    """

    def __init__(self, d: int, base: int = 10_000):
        super().__init__()

        self.d = d
        self.cos_cached = None
        self.sin_cached = None


    def _build_cache(self, x: torch.Tensor):
        """
        Compute the fixed variables that do not change during training (see recitation for more details).
        """
        B, nh, T, d = x.shape

        theta = 1 / (10000 ** (2*(torch.arange(0, d/2, 1).float()) / d)).to(x.device)

        seq_idx = torch.arange(1,T+1, device=x.device).float().to(x.device)

        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta = torch.cat((idx_theta,idx_theta),dim=1)

        self.cos_cached = idx_theta.cos()[ None, None,:, :]
        self.sin_cached = idx_theta.sin()[None,None,:, :]

    def forward(self, x: torch.Tensor):
        """
        Perform the forward pass with the input x, following equation 34 in the paper.
        """
        self._build_cache(x)
        neg_half_x= torch.cat([-x[:, :, :, self.d//2:], x[:, :, :, :self.d//2]], dim=-1)
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

        return torch.cat((x_rope, x_pass), dim=-1)
        

class CausalSelfAttention(nn.Module):
    """
    Simple Multi Headed attention. query heads = key heads = value heads
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_query_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
        self.n_head = config.n_query_head
        self.n_embd = config.n_embd
        self.rope = config.rope
        if self.rope:
            self.query_rotary_pe = RotaryPositionalEmbeddings(self.n_embd)
            self.key_rotary_pe = RotaryPositionalEmbeddings(self.n_embd)
            

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, d)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, d)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, d)

        if self.rope:
            q = self.query_rotary_pe(q)
            k = self.key_rotary_pe(k)
            
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, d) -> (B, nh, T, d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

    
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        attn_comp = self.attn(self.ln_1(x))
        x = x + attn_comp
        x = x + self.mlpf(self.ln_2(x))
        return x
    

class Model(nn.Module):
    def __init__(self, config, n_token: int, num_actions: int):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                proj=nn.Linear(n_token, config.n_embd),
                state_emb = nn.Embedding(1, n_token),
                reward_emb=nn.Embedding(2, n_token),
                action_emb=nn.Embedding(num_actions, n_token),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embd),
                head=nn.Linear(config.n_embd, num_actions, bias=False),
            )
        )
        # self.pad_tensor_states = nn.parameter.Parameter(
        #     data=torch.randn(1, 1, n_token), requires_grad=True
        # )
        # self.pad_tensor_acts = nn.parameter.Parameter(
        #     data=torch.randn(1, 1, n_token), requires_grad=True
        # )
        # self.pad_tensor_rews = nn.parameter.Parameter(
        #     data=torch.randn(1, 1, n_token), requires_grad=True
        # )
        self.n_embd = config.n_embd
        self.n_token = n_token
        self.context = []

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1))
            )
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1))
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
        for name, p in module.named_parameters():
            if name.endswith('c_proj.weight'): 
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(p.shape[-1]) / n_layer)

    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        b, t = rewards.shape[:2]
        T = rewards.shape[1] * 2 + 2
        device = rewards.device

        state_emb = self.transformer.state_emb(states)
        rew_emb = self.transformer.reward_emb(rewards)
        act_emb = self.transformer.action_emb(actions)
        # Add padding to indicate the beginning of the sequence
        # pad_states_batch = torch.tile(self.pad_tensor_states, dims=(b, 1, 1))
        # pad_acts_batch = torch.tile(self.pad_tensor_acts, dims=(b, 1, 1))
        # pad_rews_batch = torch.tile(self.pad_tensor_rews, dims=(b, 1, 1))
        # state_emb = torch.cat([pad_states_batch, state_emb], dim=1)
        # act_emb = torch.cat([pad_acts_batch, act_emb], dim=1)
        # rew_emb = torch.cat([pad_rews_batch, rew_emb], dim=1)

        sequence = (
            torch.stack([state_emb, act_emb, rew_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(b, 3 * t, self.n_token)
        )

        x = self.transformer.proj(sequence)

        for block in self.transformer.h:
            x = block(x)

        logits = self.transformer.head(self.transformer.ln_f(x))

        return logits[:, ::3]