from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import logging

def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    loss = -log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum().add(eps))
    return loss

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)

class CoxPHLoss(nn.Module):
    """Loss function for CoxPH model."""
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predictions from the model.
            targets (torch.Tensor): Target values.
        
        Returns:
            torch.Tensor: Loss value.
        """
        durations = targets[:, 0]
        events = targets[:, 1]
        
        # Check for zero events and handle accordingly
        if torch.sum(events) == 0:
            logging.warning("No events in batch, returning near zero loss")
            return torch.tensor(1e-6, dtype=preds.dtype, device=preds.device)
        
        loss = cox_ph_loss(preds, durations, events).float()
        return loss