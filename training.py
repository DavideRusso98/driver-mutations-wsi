import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import torch.nn as nn
from model import ABMIL_Multimodal
from dataset import UNIDataset
import matplotlib.pyplot as plt

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


csv_file = args.labels_file
data_dir = args.data_directory

data_frame = pd.read_csv(csv_file)

train_df, test_df = train_test_split(data_frame, test_size=0.2, random_state=SEED)

train_dataset = UNIDataset(data_frame=train_df, data_dir=data_dir, label = args.label)
#test_dataset = UNIDataset(data_frame=test_df, data_dir=data_dir, label = args.label)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
#test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

model = ABMIL_Multimodal().to(device)

learning_rate = 0.0001
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss().to(device)

loss_list = []
for e in range(EPOCHS):
    model.train()
    print(f'Start epoch: {e}')
    for i, (data, label) in tqdm(enumerate(train_loader)):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, label)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f'Loss epoch {e}: {loss.item()}')

    torch.cuda.empty_cache()

torch.save(model.state_dict(), './model_weights.pth')
print('Model saved')
plt.plot(range(len(loss_list)), loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('train loss')
plt.savefig('./train_loss.png')
with open('./loss_log.txt', 'w') as f:
    for l in loss_list:
        f.write(f'{l}\n')
print('Loss saved')
