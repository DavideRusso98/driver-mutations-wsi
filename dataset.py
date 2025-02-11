import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class UNIDataset(Dataset):
    def __init__(self, data_frame, data_dir, label, seed, transform=None, max_patches = 0):
        self.data_frame = data_frame
        self.data_dir = data_dir
        self.transform = transform
        self.label = label
        self.max_patches = max_patches
        self.seed = seed
        torch.manual_seed(seed)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        file_name = self.data_frame.iloc[idx]['slide_id']
        label = self.data_frame.iloc[idx][self.label]
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(0)

        file_path = os.path.join(self.data_dir, file_name)
        data = torch.load(file_path + '.pt', weights_only=True)

        if self.max_patches:
            n_patch = data.shape[0]
            if n_patch > self.max_patches:
                diff = n_patch - self.max_patches
                excluded_index = torch.randperm(n_patch)[:diff]
                mask = torch.ones(n_patch, dtype=torch.bool)
                mask[excluded_index] = False
                data = data[mask]


        if self.transform:
            data = self.transform(data)

        return data, label

