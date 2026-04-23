import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        if data.dim() != 1:
            raise ValueError("data must be a 1D tensor")
        if data.size(0) < context_length + 1:
            raise ValueError("data is too short for given context_length")

        torch.manual_seed(0)
        # valid starts satisfy: start + context_length < len(data)
        max_start = data.size(0) - context_length
        starts = torch.randint(low=0, high=max_start, size=(batch_size,))

        X = torch.stack([data[s : s + context_length] for s in starts], dim=0)
        Y = torch.stack([data[s + 1 : s + 1 + context_length] for s in starts], dim=0)

        return X, Y