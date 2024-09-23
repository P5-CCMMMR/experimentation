import torch
import torch.nn as nn

def RMSE(y_hat: float, y: float):
        # Add small term to avoid divison by zero
        epsilon = 1e-8
        return torch.sqrt(nn.functional.mse_loss(y_hat, y) + epsilon)
