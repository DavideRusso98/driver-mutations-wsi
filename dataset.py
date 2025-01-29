import torch
import os
from torch.utils.data import Dataset

class UNIDataset(Dataset):
    def __init__(self, data_frame, data_dir, label, transform=None):
        self.data_frame = data_frame
        self.data_dir = data_dir
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        file_name = self.data_frame.iloc[idx]['slide_id']
        label = self.data_frame.iloc[idx][self.label]
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(0)

        file_path = os.path.join(self.data_dir, file_name)
        data = torch.load(file_path + '.pt')

        if self.transform:
            data = self.transform(data)

        return data, label
