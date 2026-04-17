import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embed = nn.Embedding(vocabulary_size, 16)
        self.fc = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        # x : (B,T)
        # Return a B, 1 tensor and round to 4 decimal places
        embed = self.embed(x) # (B, T, 16)
        embed = embed.mean(dim=1) # (B, 16)
        logit = self.fc(embed) # (B, 1)
        output = self.sigmoid(logit) # (B, 1)
        return torch.round(output * 10000) / 10000
