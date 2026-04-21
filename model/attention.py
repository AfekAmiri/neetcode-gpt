import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places

        # embedded [B, T, embedding_dim] where B=batch_size, T: seq_len


        K = self.key.forward(embedded) # [B, T, C]
        Q = self.query.forward(embedded)
        V = self.value.forward(embedded)

        attention_dim = K.shape[-1]

        attention = (Q @ K.transpose(1, 2))/ attention_dim ** 0.5 # [B, T, T]

        seq_len = embedded.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=embedded.device))
        masked_scores = attention.masked_fill(mask == 0, float("-inf"))

        scores = torch.softmax(masked_scores, dim=2)

        return torch.round(scores @ V, decimals =4)

