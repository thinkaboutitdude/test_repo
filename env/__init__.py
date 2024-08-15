from gymnasium.envs.registration import register

from env.bandit import MultiArmedBanditBernoulli

register(id='MultiArmedBanditBernoulli', entry_point=MultiArmedBanditBernoulli)