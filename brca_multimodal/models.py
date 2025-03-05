import torch
from torch import nn
import torch.nn.functional as F
import math

import warnings
warnings.simplefilter("ignore", UserWarning)


class ABMIL_Multimodal(nn.Module):
    def __init__(self,
                     input_dim=1024,
                     genomics_input_dim = 19960, #19962
                     inner_dim=64, 
                     output_dim=1, 
                     use_layernorm=False,
                     genomics_dropout = 0.5,
                     dropout=0.0,
                ):
        super(ABMIL_Multimodal,self).__init__()

        self.inner_proj = nn.Linear(input_dim,inner_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(inner_dim)
        self.attention_V = nn.Linear(inner_dim, inner_dim)
        self.attention_U = nn.Linear(inner_dim, inner_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention_weights = nn.Linear(inner_dim, 1)
        
        self.genomics_dropout = nn.Dropout(genomics_dropout)
        self.fc_genomics = nn.Sequential(
                                            nn.Linear(genomics_input_dim, inner_dim),
                                            nn.ReLU(),
                                            nn.Linear(inner_dim, inner_dim),
                                        )

        final_layer_input_dim = 2*inner_dim

        self.fc1 = nn.Linear(final_layer_input_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, inner_dim//4)
        self.fc3 = nn.Linear(inner_dim//4, output_dim)
   
        
    def forward(self, data, genomics):
        x = self.inner_proj(data)
        
        if self.use_layernorm:
            x = self.layernorm(x)        
        
        # Apply attention mechanism
        V = torch.tanh(self.attention_V(x))  # Shape: (batch_size, num_patches, inner_dim)
        U = self.sigmoid(self.attention_U(x))  # Shape: (batch_size, num_patches, inner_dim)
        
        # Compute attention scores
        attn_scores = self.attention_weights(V * U)  # Shape: (batch_size, num_patches, 1)
        attn_scores = torch.softmax(attn_scores, dim=1)  # Shape: (batch_size, num_patches, 1)
        
        # Weighted sum of patch features
        weighted_sum = torch.sum(attn_scores * x, dim=1)  # Shape: (batch_size, inner_dim)
        weighted_sum = self.dropout(weighted_sum)

        wsi_embedding = weighted_sum

        genomics = self.genomics_dropout(genomics)
        genomics_embedding = self.fc_genomics(genomics)

        x = torch.cat([wsi_embedding,genomics_embedding], dim=1)


        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) # relu o tanh
        output = torch.sigmoid(self.fc3(x))
        
        return output, attn_scores


class COATT_Multimodal(nn.Module):
    def __init__(self,
                     input_dim=1024,
                     genomics_input_dim = 19960, #19962
                     inner_dim=64, 
                     output_dim=1, 
                     use_layernorm=False,
                     genomics_dropout = 0.5,
                     dropout=0.0,
                ):
        super(COATT_Multimodal,self).__init__()

        self.inner_proj = nn.Linear(input_dim,inner_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(inner_dim)
        self.attention_V = nn.Linear(inner_dim, inner_dim)
        self.attention_U = nn.Linear(inner_dim, inner_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention_weights = nn.Linear(inner_dim, 1)
        
        self.genomics_dropout = nn.Dropout(genomics_dropout)
        self.fc_genomics = nn.Sequential(
                                            nn.Linear(genomics_input_dim, inner_dim),
                                            nn.ReLU(),
                                            nn.Linear(inner_dim, inner_dim),
                                        )

        final_layer_input_dim = 2*inner_dim

        self.fc1 = nn.Linear(final_layer_input_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, inner_dim//4)
        self.fc3 = nn.Linear(inner_dim//4, output_dim)
   
        
    def forward(self, data, genomics):
        x = self.inner_proj(data)
        
        if self.use_layernorm:
            x = self.layernorm(x)        
        
        # Apply attention mechanism
        V = torch.tanh(self.attention_V(x))  # Shape: (batch_size, num_patches, inner_dim)
        U = self.sigmoid(self.attention_U(x))  # Shape: (batch_size, num_patches, inner_dim)
        
        # Compute attention scores
        attn_scores = self.attention_weights(V * U)  # Shape: (batch_size, num_patches, 1)
        attn_scores = torch.softmax(attn_scores, dim=1)  # Shape: (batch_size, num_patches, 1)
        
        # Weighted sum of patch features
        weighted_sum = torch.sum(attn_scores * x, dim=1)  # Shape: (batch_size, inner_dim)
        weighted_sum = self.dropout(weighted_sum)

        wsi_embedding = weighted_sum

        genomics = self.genomics_dropout(genomics)
        genomics_embedding = self.fc_genomics(genomics)

        x = torch.cat([wsi_embedding,genomics_embedding], dim=1)


        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) # relu o tanh
        output = torch.sigmoid(self.fc3(x))
        
        return output, attn_scores


if __name__ == '__main__':
    device = 'cuda'
    model = ABMIL_Multimodal().to(device)
    model = COATT_Multimodal().to(device)
    model.eval()
    
    data = torch.rand((300, 1024), dtype=torch.float)
    data = data.unsqueeze(0)
    data = data.to(device)

    genomics = torch.rand((1,19960), dtype=torch.float)
    genomics = genomics.unsqueeze(0)
    genomics = genomics.to(device)

    print(f'Memoria GPU prima: {torch.cuda.memory_allocated()/ (1024 ** 2)} Mbytes')
    
    out, _ = model(data, genomics)

    print(f'Memoria GPU dopo: {torch.cuda.memory_allocated()/ (1024 ** 2)} Mbytes')