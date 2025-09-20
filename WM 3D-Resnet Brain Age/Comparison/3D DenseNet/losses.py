import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedMSEMAELossWithConstraint(nn.Module):
    def __init__(self, alpha=0.5, min_age=0, max_age=120, penalty=1.0):
        """
        Combined loss with MSE + MAE and a range constraint on predictions.

        Args:
            alpha (float): Weight between MSE and MAE; larger alpha emphasizes MSE.
            min_age (float): Minimum valid age (lower bound for predictions).
            max_age (float): Maximum valid age (upper bound for predictions).
            penalty (float): Penalty weight for out-of-range predictions.
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

        # Penalty for predictions outside [min_age, max_age]
        lower_bound_penalty = torch.relu(self.min_age - predictions)
        upper_bound_penalty = torch.relu(predictions - self.max_age)
        penalty_loss = torch.mean(lower_bound_penalty + upper_bound_penalty)

        return base_loss + self.penalty * penalty_loss
