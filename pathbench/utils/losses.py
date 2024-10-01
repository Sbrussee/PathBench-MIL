from typing import Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
import logging

def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, event_weight=1.0, censored_weight=1.0, eps: float = 1e-7) -> Tensor:
    """CoxPH loss requires sorted inputs by descending duration time."""
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    
    # Calculate cumulative hazard
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    
    # Apply weights based on event or censoring
    weights = torch.where(events == 1, event_weight, censored_weight)
    
    # Compute weighted loss (use weights for events and censored samples)
    loss = -log_h.sub(log_cumsum_h).mul(events).mul(weights).sum().div(events.sum().add(eps))
    
    return loss

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, event_weight=1.0, censored_weight=1.0, eps: float = 1e-7) -> Tensor:
    """CoxPH loss that uses sorted inputs by descending duration."""
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, event_weight=event_weight, censored_weight=censored_weight, eps=eps)

class CoxPHLoss(nn.Module):
    """CoxPH loss class with event and censored weights."""
    def __init__(self, event_weight=1.0, censored_weight=1.0):
        super().__init__()
        self.event_weight = event_weight
        self.censored_weight = censored_weight

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predictions from the model.
            targets (torch.Tensor): Target values containing durations and event indicators.
        
        Returns:
            torch.Tensor: Weighted loss value.
        """
        durations = targets[:, 0]
        events = targets[:, 1]
        
        # Check for zero events and handle accordingly
        if torch.sum(events) == 0:
            return torch.tensor(1e-6, dtype=preds.dtype, device=preds.device)
        
        # Calculate log_h from preds
        loss = cox_ph_loss(preds, durations, events, self.event_weight, self.censored_weight).float()
        return loss

class RankingLoss(nn.Module):
    """Pairwise ranking loss for survival analysis with weights for events and censored data."""
    def __init__(self, margin=1.0, event_weight=1.0, censored_weight=1.0):
        super().__init__()
        self.margin = margin
        self.event_weight = event_weight
        self.censored_weight = censored_weight

    def forward(self, preds, targets):
        durations = targets[:, 0]
        events = targets[:, 1]
        
        loss = 0.0
        num_pairs = 0

        # Compare all pairs of individuals
        for i in range(len(durations)):
            for j in range(len(durations)):
                if durations[i] < durations[j] and events[i] == 1:
                    # Apply event_weight and censored_weight
                    weight = self.event_weight if events[i] == 1 else self.censored_weight
                    # Margin ranking loss
                    loss += weight * torch.relu(self.margin - (preds[i] - preds[j]))
                    num_pairs += 1

        if num_pairs > 0:
            loss /= num_pairs

        return loss

class DeepHitLoss(nn.Module):
    """DeepHit loss combining likelihood and ranking loss with weights."""
    def __init__(self, alpha=0.5, event_weight=1.0, censored_weight=1.0):
        super().__init__()
        self.alpha = alpha  # Weight between likelihood and ranking loss
        self.event_weight = event_weight
        self.censored_weight = censored_weight

    def forward(self, preds, targets):
        durations = targets[:, 0]
        events = targets[:, 1]
        num_time_intervals = preds.shape[1]
        
        # Scale durations to the number of time intervals
        duration_indices = (durations / durations.max() * (num_time_intervals - 1)).long()

        # Likelihood loss (for the event times)
        idx = torch.arange(preds.size(0))
        likelihood_loss = -torch.log(preds[idx, duration_indices] + 1e-8) * events

        # Apply event and censoring weights
        weights = torch.where(events == 1, self.event_weight, self.censored_weight)
        likelihood_loss = likelihood_loss.mul(weights)

        # Ranking loss: pairwise comparisons between individuals
        rank_loss = 0.0
        count = 0
        for i in range(preds.size(0)):
            for j in range(preds.size(0)):
                if durations[i] < durations[j] and events[i] == 1:
                    rank_loss += torch.relu(preds[j, duration_indices[i]] - preds[i, duration_indices[i]]) * self.event_weight
                    count += 1

        rank_loss = rank_loss / count if count > 0 else 0.0

        # Combine likelihood and ranking losses
        return torch.mean(self.alpha * likelihood_loss + (1 - self.alpha) * rank_loss)
        
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[Tensor] = None):
        """
        Cross-entropy loss. This is the standard loss function for multi-class classification problems.
        Args:
            weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
        """
        super().__init__()
        self.require_attention = False
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(preds.float(), targets.float())


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean', weight: Optional[Tensor] = None):
        """
        Focal loss for imbalanced classification problems. Focal loss is designed to address class imbalance by
        down-weighting the loss assigned to well-classified examples. 
        Args:
            alpha (float): Scaling factor for positive examples.
            gamma (float): Focusing parameter.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
        """
        super().__init__()
        self.require_attention = False
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)
        
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(preds, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, weight: Optional[Tensor] = None):
        """
        Cross-entropy loss with label smoothing. Label smoothing is a regularization technique that
        encourages the model to be less confident about its predictions.
        Args:
            smoothing (float): Amount of label smoothing to apply.
            weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
        """
        super().__init__()
        self.require_attention = False
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.weight = weight
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        assert preds.size(0) == targets.size(0)

        log_preds = torch.log_softmax(preds, dim=-1)
        n_classes = preds.size(-1)

        with torch.no_grad():
            # If targets are one-hot encoded, convert them to class indices
            if targets.dim() > 1:
                targets = torch.argmax(targets, dim=1)

            # Ensure targets are of type int64 for the scatter_ operation
            targets = targets.to(torch.int64)

            # Create a tensor for the true distribution
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))

            # Scatter the confidence into the true distribution
            true_dist.scatter_(1, targets.view(-1, 1), self.confidence)

        # Apply class weights if provided
        if self.weight is not None:
            true_dist *= self.weight[targets].view(-1, 1)

        return self.criterion(log_preds, true_dist)

class CrossEntropyWithEntropyMinimizationLoss(nn.Module):
    def __init__(self, weight_entropy: float = 1.0, weight: Optional[Tensor] = None):
        """
        Cross-entropy loss with entropy minimization regularization.

        Args:
            weight_entropy (float): Weight of the entropy minimization loss.
            weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
        """
        super().__init__()
        self.require_attention = False
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight)
        self.weight_entropy = weight_entropy

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        # Calculate cross-entropy loss
        cross_entropy_loss = self.cross_entropy_loss(preds, targets)

        # Convert logits to probabilities using softmax
        if preds.dim() > 1 and preds.size(1) > 1:  # Multi-class case
            probs = torch.softmax(preds, dim=1)
        else:  # Binary case
            probs = torch.sigmoid(preds)

        # Calculate entropy minimization loss
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        entropy_minimization_loss = self.weight_entropy * entropy.mean()

        # Combine the two losses
        total_loss = cross_entropy_loss + entropy_minimization_loss

        return total_loss

class AttentionEntropyMinimizedCrossEntropyLoss(nn.Module):
    def __init__(self, entropy_lambda: float = 1.0, weight: Optional[Tensor] = None):
        """
        Entropy minimization loss for attention mechanisms.
        Args:
            entropy_lambda (float): Regularization strength for the entropy minimization.
            weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
        """
        super().__init__()
        self.require_attention = True
        self.entropy_lambda = entropy_lambda
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: Tensor, targets: Tensor, attention_weights: Tensor) -> Tensor:
        # Standard cross-entropy loss
        ce_loss = self.cross_entropy(preds, targets)

        # Check if attention weights are normalized
        if attention_weights.dim() > 1:
            attention_weights = torch.softmax(attention_weights, dim=-1)

        # Entropy minimization term
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=1)
        entropy_min_reg = torch.mean(entropy)
        
        # Total loss
        loss = ce_loss + self.entropy_lambda * entropy_min_reg
        return loss

class DiversityRegularizedCrossEntropyLoss(nn.Module):
    def __init__(self, diversity_lambda: float = 1.0, weight: Optional[Tensor] = None):
        """
        Diversity regularization loss for attention mechanisms. This loss encourages the attention weights to be diverse.

        Args:
            diversity_lambda (float): Regularization strength for the diversity regularization.
            weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
        """
        super().__init__()
        self.require_attention = True
        self.diversity_lambda = diversity_lambda
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: Tensor, targets: Tensor, attention_weights: Tensor) -> Tensor:
        # Standard cross-entropy loss
        ce_loss = self.cross_entropy(preds, targets)

        # Check if attention weights are normalized
        if attention_weights.dim() > 1:
            attention_weights = torch.softmax(attention_weights, dim=-1)

        # Diversity regularization term
        variance = torch.var(attention_weights, dim=1, unbiased=False)
        diversity_reg = -torch.mean(variance)
        
        # Total loss
        loss = ce_loss + self.diversity_lambda * diversity_reg
        return loss

class SparseAttentionCrossEntropyLoss(nn.Module):
    def __init__(self, attention_lambda: float = 1.0, weight: Optional[Tensor] = None):
        """
        Sparse attention regularization loss for attention mechanisms. This loss encourages the attention weights to be sparse.

        Args:
            attention_lambda (float): Regularization strength for the attention sparsity.
            weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
        """
        super().__init__()
        self.require_attention = True
        self.attention_lambda = attention_lambda
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: Tensor, targets: Tensor, attention_weights: Tensor) -> Tensor:
        # Standard cross-entropy loss
        ce_loss = self.cross_entropy(preds, targets)

        # Check if attention weights are normalized
        if attention_weights.dim() > 1:
            attention_weights = torch.softmax(attention_weights, dim=-1)

        # Sparse attention regularization term
        sparse_reg = torch.sum(attention_weights ** 2)
        
        # Total loss
        loss = ce_loss + self.attention_lambda * sparse_reg
        return loss
