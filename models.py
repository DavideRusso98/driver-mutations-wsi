import torch
from torch import nn

import warnings
warnings.simplefilter("ignore", UserWarning)

class ABMIL(nn.Module):
    def __init__(self,
                     input_dim=1024,
                     inner_dim=64, 
                     output_dim=1, 
                     use_layernorm=False, 
                     dropout=0.0,
                ):
        super(ABMIL,self).__init__()

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

        self.fc1 = nn.Linear(inner_dim, inner_dim//4)
        self.fc2 = nn.Linear(inner_dim//4, output_dim)
        #self.fc2 = nn.Linear(inner_dim, output_dim)
   
        
    def forward(self, data):
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

        # Final WSI embedding
        x = weighted_sum

        x = torch.tanh(self.fc1(x)) # relu o tanh
        output = torch.sigmoid(self.fc2(x))
        
        return output, attn_scores


if __name__ == '__main__':
    device = 'cuda'
    model = ABMIL().to(device)
    model.eval()
    
    data = torch.load('/home/fabio/Documenti/Università/Magistrale/Secondo Anno/Primo Semestre/AI for Bioinformatics/Progetto/TCGA/TCGA_BRCA/Data/wsi/features_UNI/pt_files/TCGA-3C-AALK-01A-01-TSA.B64ED65E-C91A-42C9-89A5-1B099C7112C3.pt')
    data = data.unsqueeze(0)
    data = data.to(device)

    print(f'Memoria GPU prima: {torch.cuda.memory_allocated()/ (1024 ** 2)} Mbytes')
    
    out = model(data)

    print(f'Memoria GPU dopo: {torch.cuda.memory_allocated()/ (1024 ** 2)} Mbytes')

    print(out.shape)