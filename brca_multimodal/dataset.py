import torch
import os
import pandas as pd
from torch.utils.data import Dataset

###########

import numpy as np
from PIL import Image



class UNIDataset(Dataset):
    def __init__(self, gene_expr, data_frame, data_dir, label, seed, transform=None, max_patches = 0):
        self.gene_expr = gene_expr
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
        
        # genomic data
        case_id = self.data_frame.iloc[idx]['case_id']
        genes = self.gene_expr.loc[self.gene_expr['case_id'] == case_id].iloc[0, 1:]
        genes = torch.tensor(genes, dtype=torch.float32)#.unsqueeze(0) 

        if self.transform:
            data = self.transform(data)

            genes = self.transform(genes)

        return data, genes, label

