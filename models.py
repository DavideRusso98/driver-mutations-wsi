import torch
from torch import nn
import torch.nn.functional as F
import math

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



class MSA(nn.Module):
  def __init__(self, input_dim, embed_dim, num_heads):
    '''
    input_dim: Dimension of input token embeddings
    embed_dim: Dimension of internal key, query, and value embeddings
    num_heads: Number of self-attention heads
    '''

    super().__init__()

    self.input_dim = input_dim
    self.embed_dim = embed_dim
    self.num_heads = num_heads

    self.K_embed = nn.Linear(input_dim, embed_dim, bias=False)
    self.Q_embed = nn.Linear(input_dim, embed_dim, bias=False)
    self.V_embed = nn.Linear(input_dim, embed_dim, bias=False)
    self.out_embed = nn.Linear(embed_dim, embed_dim, bias=False)

  def forward(self, x):
    '''
    x: input of shape (batch_size, max_length, input_dim)
    return: output of shape (batch_size, max_length, embed_dim)
    '''

    batch_size, max_length, given_input_dim = x.shape
    assert given_input_dim == self.input_dim
    assert self.embed_dim % self.num_heads == 0

    # compute K, Q, V
    K = self.K_embed(x) # (batch_size, max_length, embed_dim)
    Q = self.Q_embed(x)
    V = self.V_embed(x)

    # TODO: split each KQV into heads, by reshaping each into (batch_size, max_length, self.num_heads, indiv_dim)
    indiv_dim = self.embed_dim // self.num_heads
    K = K.reshape(batch_size, max_length, self.num_heads, indiv_dim)# TODO
    Q = Q.reshape(batch_size, max_length, self.num_heads, indiv_dim)
    V = V.reshape(batch_size, max_length, self.num_heads, indiv_dim)

    K = K.permute(0, 2, 1, 3) # (batch_size, num_heads, max_length, embed_dim / num_heads)
    Q = Q.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)

    K = K.reshape(batch_size * self.num_heads, max_length, indiv_dim)
    Q = Q.reshape(batch_size * self.num_heads, max_length, indiv_dim)
    V = V.reshape(batch_size * self.num_heads, max_length, indiv_dim)

    # transpose and batch matrix multiply
    K_T = K.permute(0, 2, 1) # This is our K transposed so we can do a simple batched matrix multiplication (see torch.bmm for more details and the quick solution)
    QK = torch.matmul(Q,K_T)# TODO: Compute the weights before dividing by square root of d (batch_size * num_heads, max_length, max_length)

    # calculate weights by dividing everything by the square root of d (self.embed_dim)
    weights = QK/math.sqrt(self.embed_dim)# TODO
    weights = F.softmax(weights, dim=-1)# TODO Take the softmax over the last dimension (see torch.functional.Softmax) (batch_size * num_heads, max_length, max_length)

    w_V = torch.matmul(weights,V)# weights is (batch_size * num_heads, max_length, max_length) and V is (batch_size * self.num_heads, max_length, indiv_dim), so we want the matrix product of weights @ V

    # rejoin heads
    w_V = w_V.reshape(batch_size, self.num_heads, max_length, indiv_dim)
    w_V = w_V.permute(0, 2, 1, 3) # (batch_size, max_length, num_heads, embed_dim / num_heads)
    w_V = w_V.reshape(batch_size, max_length, self.embed_dim)

    out = self.out_embed(w_V)

    return out, weights.detach()
    


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, batch_first=True):
        super(TransformerBlock, self).__init__()
        self.att = MSA(embed_dim, embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(),nn.Linear(ff_dim, embed_dim)
        )
       
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        batch_size = inputs.size(dim=0)

        attn_output, scores = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        output = self.layernorm2(out1 + ffn_output)
        return output, scores



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