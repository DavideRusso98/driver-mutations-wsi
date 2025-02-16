import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split,StratifiedKFold
from torch.optim import Adam, RAdam
import torch.nn as nn
import sys
from dataset_generic import *
from dataset import UNIDataset
from clam import * 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='BRCA overexprection')
parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed')
parser.add_argument('--data_directory', type=str, default='/work/ai4bio2024/brca_surv/LAB_WSI_Genomics/TCGA_BRCA/Data/wsi/features_UNI/pt_files',
                        help='Dataset directory')
parser.add_argument('--labels_file', type=str, default='/work/ai4bio2024/brca_surv/dataset/dataset_brca_wsi.csv',
                        help='label file path')
parser.add_argument('--label', type=str, default='BRCA1',
                        help='Label to use for training')
parser.add_argument('--epoch', '-e', type=int, default=10,
                    help='Number of epochs')
parser.add_argument('--folds', '-f', type=int, default=5,
                    help='Number of folds')

args = parser.parse_args()

SEED = args.seed
EPOCHS = args.epoch
FOLDS = args.folds
MODEL = "clam"
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

csv_file = args.labels_file
data_dir = args.data_directory

data_frame = pd.read_csv(csv_file)

#####################################################################
def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

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

def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    running_loss = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)

        loss = loss_fn(logits, logits)
        

        running_loss += loss.item()
        

        # Aggiornamento pesi
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Taining Loss: {epoch_loss:.4f}')

    torch.cuda.empty_cache()
        # Debugging output
    #     if batch_idx % 10 == 0:
    #         print(f"Batch {batch_idx} - Loss: {loss_value:.4f} Real: {label.tolist()}")

    # # Media degli errori
    # train_loss /= len(loader)
    # train_error /= len(loader)

    # print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    # torch.cuda.empty_cache()

'''test_size=0.15
val_size=0.15
train_val_df, test_df = train_test_split(data_frame, test_size=test_size, stratify = data_frame[args.label], random_state=SEED)
val_size_adjusted = val_size / (1 - test_size)
train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, stratify = train_val_df[args.label], random_state=SEED)'''

train_val_df, test_df = train_test_split(data_frame, test_size=0.2, stratify = data_frame[args.label], random_state=SEED)
y = train_val_df[args.label]
X = train_val_df.drop(args.label, axis = 1)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for f, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f'\n#########  Fold {f+1}/{FOLDS}  #########\n')
    train_dataset = UNIDataset(data_frame=train_val_df.iloc[train_index], data_dir=data_dir, label = args.label, seed=SEED, max_patches=4096)
    val_dataset = UNIDataset(data_frame=train_val_df.iloc[val_index], data_dir=data_dir, label = args.label, seed=SEED, max_patches=4096)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)


    loss_fn = nn.BCEWithLogitsLoss().to(device)
    model = CLAM_SB(n_classes=1,instance_loss_fn=loss_fn).to(device)

    LR = 0.0001
    WEIGHT_DECAY = 0.0001
    NUM_ACCUMULATION_STEPS = 8
    PATIENCE = 5
    optimizer = RAdam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)



    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(EPOCHS):

        train_loop_clam(epoch, model, train_loader, optimizer, 2, 0, None, loss_fn)


        #Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, label in tqdm(val_loader):
                data = data.to(device)
                logits, Y_prob, Y_hat, _, _ = model(data)
                loss = loss_fn(logits, logits)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}')

        if early_stopping(val_loss):
            print("Early stopped")
            break

    torch.save(model.state_dict(), f'./models/{MODEL}_model_weights_{f+1}.pth')

print('Done!')

