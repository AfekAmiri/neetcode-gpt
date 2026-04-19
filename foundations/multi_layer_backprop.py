import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)


        # Convert inputs to numpy arrays
        x = np.array(x, dtype=np.float64)           # (in_dim,)
        W1 = np.array(W1, dtype=np.float64)         # (in_dim, hidden_dim)
        b1 = np.array(b1, dtype=np.float64)         # (hidden_dim,)
        W2 = np.array(W2, dtype=np.float64)         # (hidden_dim, out_dim)
        b2 = np.array(b2, dtype=np.float64)         # (out_dim,)
        y_true = np.array(y_true, dtype=np.float64) # (out_dim,)

        # Forward
        z1 = x @ np.transpose(W1) + b1                    # (hidden_dim,)
        a1 = np.maximum(0, z1)              # ReLU
        y_pred = a1 @ np.transpose(W2) + b2               # (out_dim,)

        # Loss (MSE)
        diff = y_pred - y_true
        loss = np.mean(diff ** 2)

        # Backward
        out_dim = y_true.shape[0]
        dy_pred = (2.0 / out_dim) * diff    # d(MSE)/d(y_pred), shape (out_dim,)

        dW2 = np.outer(dy_pred, a1)         # (hidden_dim, out_dim)
        db2 = dy_pred                        # (out_dim,)

        da1 = dy_pred @ W2      # (hidden_dim,)
        dz1 = da1 * (z1 > 0).astype(np.float64)  # ReLU grad

        dW1 = np.outer(dz1, x)               # (in_dim, hidden_dim)
        db1 = dz1                            # (hidden_dim,)

        return {
            "loss": round(float(loss), 4),
            "dW1": (np.round(dW1, 4)+0.0).tolist(),
            "db1": (np.round(db1, 4)+0.0).tolist(),
            "dW2": (np.round(dW2, 4)+0.0).tolist(),
            "db2": (np.round(db2, 4)+0.0).tolist(),
        }
