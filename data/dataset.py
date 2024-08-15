import numpy as np

from torch.utils.data.dataloader import Dataset

class SequenceData(Dataset):
    def __init__(self, data_path: str, context_len: int):
        super().__init__()
        self.data = np.load(data_path)['data']
        # print(f'self data - {self.data.shape}')
        # print(self.data[8, -6:])
        # i1, i2 = 7, 667
        # print(self.data[i1, 3 * i2:3 * (i2 + 20):3])
        self.context_len = context_len
        # print(f'context len - {self.context_len}')

    def __len__(self) -> int:
        return self.data.shape[0] * (self.data.shape[1] // 3 - self.context_len + 1)
    
    def __getitem__(self, index) -> np.ndarray:
        # print(f'index - {index}')
        num_hists = self.data.shape[0]
        i1, i2 = index % num_hists, index // num_hists
        # print(f'data shape - {self.data.shape}')
        # print(i1, i2)
        states = self.data[i1, 3 * i2:3 * (i2 + self.context_len):3]
        actions = self.data[i1, 3 * i2 + 1:3 * (i2 + self.context_len) + 1:3]
        rewards = self.data[i1, 3 * i2 + 2:3 * (i2 + self.context_len) + 2:3]
        # print(f'shapes - {states.shape} - {actions.shape}, {rewards.shape}')
        return states, actions, rewards