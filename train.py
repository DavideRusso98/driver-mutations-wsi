import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import Adam, RAdam
import torch.nn as nn
from models import *
from dataset import UNIDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='BRCA overexprection')
parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed')
parser.add_argument('--data_directory', type=str, default='/work/ai4bio2024/brca_surv/LAB_WSI_Genomics/TCGA_BRCA/Data/wsi/features_UNI/pt_files',
                        help='Dataset directory')
parser.add_argument('--labels_file', type=str, default='/work/ai4bio2024/brca_surv/dataset/dataset_brca.csv',
                        help='label file path')
parser.add_argument('--label', type=str, default='BRCA1',
                        help='Label to use for training')
parser.add_argument('--epoch', '-e', type=int, default=10,
                    help='Number of epochs')

args = parser.parse_args()

SEED = args.seed
EPOCHS = args.epoch

def setup(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA.
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup(SEED)

if torch.cuda.is_available():
    device = 'cuda'
    print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'
    print('No GPU!')

print(f'Predicting {args.label} overexprection')


#####################################################################à

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            if self.verbose:
                print(f'New best loss: {self.best_loss}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        return False



csv_file = args.labels_file
data_dir = args.data_directory

data_frame = pd.read_csv(csv_file)


'''test_size=0.15
val_size=0.15
train_val_df, test_df = train_test_split(data_frame, test_size=test_size, stratify = data_frame[args.label], random_state=SEED)
val_size_adjusted = val_size / (1 - test_size)
train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, stratify = train_val_df[args.label], random_state=SEED)'''

train_df, test_df = train_test_split(data_frame, test_size=0.2, stratify = data_frame[args.label], random_state=SEED)

train_dataset = UNIDataset(data_frame=train_df, data_dir=data_dir, label = args.label)
#test_dataset = UNIDataset(data_frame=test_df, data_dir=data_dir, label = args.label)
#val_dataset = UNIDataset(data_frame=val_df, data_dir=data_dir, label = args.label)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
#test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
#val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)

model = ABMIL(use_layernorm=True).to(device)
#model = MHMIL(use_layernorm=True).to(device)

LR = 0.001
WEIGHT_DECAY = 0.0001
optimizer = RAdam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCELoss().to(device)

early_stopping = EarlyStopping(patience=10, verbose=True)

for e in range(EPOCHS):
    model.train()
    running_loss= 0.0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {e + 1}/{EPOCHS}, Taining Loss: {epoch_loss:.4f}')

    torch.cuda.empty_cache()

    '''#Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, label in tqdm(val_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch {e+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}')

    if early_stopping(val_loss):
        print("Early stopped")
        break'''

torch.save(model.state_dict(), './model_weights.pth')
print('Model saved')
