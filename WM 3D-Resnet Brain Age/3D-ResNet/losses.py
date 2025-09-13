"""
Hybrid MSE+MAE Loss with Age-Range Constraint for Brain Age Regression

@author: Puzhen & Ruijia
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedMSEMAELossWithConstraint(nn.Module):
    def __init__(self, alpha=0.5, min_age=0, max_age=120, penalty=1.0):
        """
        Combines MSE and MAE with range constraints on predictions.
        """
        super(CombinedMSEMAELossWithConstraint, self).__init__()
        self.alpha = alpha
        self.min_age = min_age
        self.max_age = max_age
        self.penalty = penalty

    def forward(self, predictions, targets):
        mse_loss = F.mse_loss(predictions, targets)
        mae_loss = F.l1_loss(predictions, targets)
        base_loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss

        lower_bound_penalty = torch.relu(self.min_age - predictions)
        upper_bound_penalty = torch.relu(predictions - self.max_age)
        penalty_loss = torch.mean(lower_bound_penalty + upper_bound_penalty)

        total_loss = base_loss + self.penalty * penalty_loss
        return total_loss
