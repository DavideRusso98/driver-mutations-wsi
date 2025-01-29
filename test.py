import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import ABMIL_Multimodal
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

args = parser.parse_args()

SEED = args.seed

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

print(f'Testing {args.label} overexprection')


csv_file = args.labels_file
data_dir = args.data_directory

data_frame = pd.read_csv(csv_file)

train_df, test_df = train_test_split(data_frame, test_size=0.2, random_state=SEED)

#train_dataset = UNIDataset(data_frame=train_df, data_dir=data_dir, label = args.label)
test_dataset = UNIDataset(data_frame=test_df, data_dir=data_dir, label = args.label)

#train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

model = ABMIL_Multimodal()
model.load_state_dict(torch.load('./model_weights.pth'))
model = model.to(device)
model.eval()

with torch.no_grad():
    y_pred_list = []
    y_test_list = []
    for data, label in tqdm(test_loader):
        data = data.to(device)
        label = label.to(device)
        outputs = model(data)
        y_pred_list.append(outputs)
        y_test_list.append(label)

    y_pred_tensor = torch.cat(y_pred_list)
    y_test_tensor = torch.cat(y_test_list)
    y_pred_class = (y_pred_tensor > 0.5).float()

# Calcola l'accuratezza
accuracy = (y_pred_class.eq(y_test_tensor).sum() / y_test_tensor.size(0)).item()
print(f'Accuracy: {accuracy:.4f}')
