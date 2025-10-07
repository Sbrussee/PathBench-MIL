import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple


### Helpers
def _ce_targets(targets: Tensor) -> Tensor:
    """
    Normalize any classification targets to Long indices of shape (N,).
    Handles:
      - one-hot/prob vectors (float/bool) -> argmax
      - float indices -> round then cast
      - int/bool tensors with trailing singleton dims -> squeeze
    """
    if targets is None:
        raise ValueError("targets is None")

    # (N, 1) -> (N,)
    if targets.ndim > 1 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)

    # One-hot (bool or float) => argmax to indices
    if (targets.ndim > 1) and (targets.size(-1) > 1):
        return targets.argmax(dim=-1).to(torch.long).view(-1)

    # Scalar per-sample targets
    if torch.is_floating_point(targets):
        # floats that are actually indices
        return targets.round().to(torch.long).view(-1)

    if targets.dtype == torch.bool:
        # bool class ids 0/1
        return targets.to(torch.long).view(-1)

    # ints already
    return targets.to(torch.long).view(-1)


def _float_logits(preds: Tensor) -> Tensor:
    return preds.float()

################################################################################
# 1) CLASSIFICATION LOSSES (unchanged, no special signature constraints)
################################################################################

class CrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy classification loss.

    forward(preds, targets)
    """
    def __init__(self, weight: Optional[Tensor] = None):
        super().__init__()
        self.require_attention = False
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(_float_logits(preds), _ce_targets(targets))


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced classification.

    forward(preds, targets)
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 reduction: str = 'mean', weight: Optional[Tensor] = None):
        super().__init__()
        self.require_attention = False
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(_float_logits(preds), _ce_targets(targets))
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Label smoothing cross-entropy for classification. 

    forward(preds, targets)
    """
    def __init__(self, smoothing: float = 0.1, weight: Optional[Tensor] = None):
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
            if targets.dim() > 1:
                targets = torch.argmax(targets, dim=1)
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.view(-1, 1), self.confidence)

        if self.weight is not None:
            true_dist *= self.weight[targets].view(-1, 1)
        return self.criterion(log_preds, true_dist)


class CrossEntropyWithEntropyMinimizationLoss(nn.Module):
    """
    Cross-entropy + prediction-entropy minimization for classification. (Local/custom.)
    
    forward(preds, targets)
    """
    def __init__(self, weight_entropy: float = 1.0, weight: Optional[Tensor] = None):
        super().__init__()
        self.require_attention = False
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight)
        self.weight_entropy = weight_entropy

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        ce_loss = self.cross_entropy_loss(_float_logits(preds), _ce_targets(targets))
        if preds.dim() > 1 and preds.size(1) > 1:
            probs = torch.softmax(preds, dim=1)
        else:
            probs = torch.sigmoid(preds)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        return ce_loss + self.weight_entropy * entropy.mean()


class AttentionEntropyMinimizedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy + attention entropy minimization. (Local/custom, classification only.)

    forward(preds, targets, attention_weights)
    """
    def __init__(self, entropy_lambda: float = 1.0, weight: Optional[Tensor] = None):
        super().__init__()
        self.require_attention = True
        self.entropy_lambda = entropy_lambda
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: Tensor, targets: Tensor, attention_weights: Tensor) -> Tensor:
        ce_loss = self.cross_entropy(_float_logits(preds), _ce_targets(targets))
        if attention_weights.dim() > 1:
            attention_weights = torch.softmax(attention_weights, dim=-1)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=1)
        return ce_loss + self.entropy_lambda * entropy.mean()


class DiversityRegularizedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy + attention diversity penalty (variance). (Local/custom.)

    forward(preds, targets, attention_weights)
    """
    def __init__(self, diversity_lambda: float = 1.0, weight: Optional[Tensor] = None):
        super().__init__()
        self.require_attention = True
        self.diversity_lambda = diversity_lambda
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: Tensor, targets: Tensor, attention_weights: Tensor) -> Tensor:
        ce_loss = self.cross_entropy(_float_logits(preds), _ce_targets(targets))
        if attention_weights.dim() > 1:
            attention_weights = F.softmax(attention_weights, dim=-1)
        var_ = torch.var(attention_weights, dim=1, unbiased=False)
        diversity_reg = -var_.mean()
        return ce_loss + self.diversity_lambda * diversity_reg


class SparseAttentionCrossEntropyLoss(nn.Module):
    """
    Cross-entropy + attention sparsity (L2). (Local/custom classification.)

    forward(preds, targets, attention_weights)
    """
    def __init__(self, attention_lambda: float = 1.0, weight: Optional[Tensor] = None):
        super().__init__()
        self.require_attention = True
        self.attention_lambda = attention_lambda
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: Tensor, targets: Tensor, attention_weights: Tensor) -> Tensor:
        ce_loss = self.cross_entropy(_float_logits(preds), _ce_targets(targets))
        if attention_weights.dim() > 1:
            attention_weights = F.softmax(attention_weights, dim=-1)
        sparse_reg = torch.sum(attention_weights ** 2)
        return ce_loss + self.attention_lambda * sparse_reg


###############################################################################
# Regression Losses (preds, targets)
###############################################################################

class MSELossReg(nn.Module):
    """
    Mean Squared Error loss for regression tasks.

    Signature:
        forward(self, preds, targets)

      - preds: [B, *]
      - targets: [B, *] with the same shape
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.mse(preds, targets)


class L1LossReg(nn.Module):
    """
    L1 (Mean Absolute Error) loss for regression tasks.

    Signature:
        forward(self, preds, targets)

      - preds: [B, *]
      - targets: [B, *]
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.l1(preds, targets)


class HuberLossReg(nn.Module):
    """
    Huber (Smooth L1) loss for regression tasks.
    Less sensitive to outliers than MSE.

    Signature:
        forward(self, preds, targets)

      - preds: [B, *]
      - targets: [B, *]
    """
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.huber = nn.SmoothL1Loss(beta=delta, reduction=reduction)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.huber(preds, targets)


# ------------------- SURVIVAL LOSSES (modified to take (preds, targets)) ------------------- #

# We import the underlying PyCox implementations
from pycox.models.loss import (
    cox_ph_loss as pycox_cox_ph_loss,
    nll_logistic_hazard as pycox_nll_logistic_hazard,
    nll_pmf as pycox_nll_pmf,
    nll_mtlr as pycox_nll_mtlr,
    nll_pc_hazard_loss as pycox_nll_pc_hazard_loss,
    bce_surv_loss as pycox_bce_surv_loss,
)

class SurvivalLossBase(nn.Module):
    """Optional interface for survival losses."""
    index_base: int = 0      # 0- or 1-based labels
    link: str | None = None  # e.g. "logit", "cloglog", None if not hazards

    @torch.no_grad()
    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        """Return scalar risk per sample; higher = higher event risk."""
        raise NotImplementedError
    

################################################################################
# CoxPHLoss
################################################################################

class CoxPHLoss(SurvivalLossBase):
    """
    Cox PH loss from pycox (continuous-time survival).

    Args:
        event_weight: not used directly here, retained for potential extension
        censored_weight: not used directly here, retained for potential extension

    forward(self, preds, targets):
        - preds: shape [N] (log hazards)
        - targets: shape [N, 2], where
            durations = targets[:, 0]
            events    = targets[:, 1]
    """
    def __init__(self, event_weight=1.0, censored_weight=1.0):
        super().__init__()
        self.event_weight = event_weight
        self.censored_weight = censored_weight

    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        return scores.view(-1).float()
    
    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0]
        events = targets[:, 1].bool()
        log_h = preds.view(-1)  # same as old log_h
        durations = durations.view(-1)
        events = events.view(-1)
        # Use pycox
        return pycox_cox_ph_loss(log_h, durations, events)


################################################################################
# ExponentialConcordanceLoss
################################################################################

class ExponentialConcordanceLoss(nn.Module):
    def __init__(self, event_weight=1.0, censored_weight=1.0):
        super().__init__()
        self.event_weight = event_weight
        self.censored_weight = censored_weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0]
        events = targets[:, 1].bool()
        preds = preds.view(-1)

        finite = torch.isfinite(durations) & torch.isfinite(preds)
        if finite.sum() < 2:
            return preds.sum() * 0.0

        durations = durations[finite]
        events    = events[finite]
        preds     = preds[finite]

        n = preds.shape[0]
        dur_i = durations.unsqueeze(0).repeat(n, 1)
        dur_j = durations.unsqueeze(1).repeat(1, n)
        evt_i = events.unsqueeze(0).repeat(n, 1)

        final_mask = (dur_i < dur_j) & evt_i
        if not final_mask.any():
            return preds.sum() * 0.0

        p_i = preds.unsqueeze(0).repeat(n, 1)
        p_j = preds.unsqueeze(1).repeat(1, n)

        pred_diff = p_i - p_j
        loss_matrix = torch.exp(-pred_diff) * final_mask.float()

        return loss_matrix.mean()


################################################################################
# RankingLoss
################################################################################

class RankingLoss(nn.Module):
    def __init__(self, margin=1.0, event_weight=1.0, censored_weight=1.0):
        super().__init__()
        self.margin = margin
        self.event_weight = event_weight
        self.censored_weight = censored_weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0]
        events = targets[:, 1].bool()

        preds = preds.view(-1)
        # mask non-finite to avoid degenerate batches
        finite = torch.isfinite(durations) & torch.isfinite(preds)
        if finite.sum() < 2:
            return preds.sum() * 0.0  # keep graph

        durations = durations[finite]
        events    = events[finite]
        preds     = preds[finite]

        # pairwise masks
        dur_i = durations[:, None]
        dur_j = durations[None, :]
        evt_i = events[:, None]                   # event must occur in i
        mask  = (dur_i < dur_j) & evt_i

        if not mask.any():
            return preds.sum() * 0.0  # keep graph

        diff = preds[:, None] - preds[None, :]
        loss_mat = torch.relu(self.margin - diff)[mask].float()
        return loss_mat.mean()


################################################################################
# NLLLogisticHazardLoss
################################################################################

class NLLLogisticHazardLoss(SurvivalLossBase):
    """
    Discrete-time survival loss based on Logistic Hazard parameterization (PyCox).

    forward(self, preds, targets):
      - preds: Tensor of shape [N, T] (logits for hazard(t))
      - targets: shape [N, 2]
        durations = targets[:, 0]
        events    = targets[:, 1]
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.link = "logit"
        
    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        h = torch.sigmoid(scores)
        return -torch.sum(torch.log1p(-h), dim=1)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0].long()
        events = targets[:, 1].bool()
        return pycox_nll_logistic_hazard(preds, durations, events, self.reduction)


################################################################################
# NLLPMFLoss
################################################################################

class NLLPMFLoss(SurvivalLossBase):
    """
    Discrete-time survival loss with a PMF parameterization (PyCox).

    forward(self, preds, targets):
      - preds: [N, T], real-valued (converted via softmax to PMF).
      - targets: shape [N, 2]
        durations = targets[:, 0]
        events    = targets[:, 1]
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        pmf = torch.softmax(scores, dim=-1)
        eps = 1e-9
        pmf = pmf.clamp(eps, 1-eps)
        N, K = pmf.shape
        S_prev = torch.ones(N, device=scores.device, dtype=scores.dtype)
        H = torch.zeros(N, device=scores.device, dtype=scores.dtype)
        for k in range(K):
            h_k = (pmf[:, k] / S_prev).clamp(eps, 1 - eps)
            H += -torch.log1p(-h_k)
            S_prev *= (1 - h_k)
        return H


    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0].long()
        events = targets[:, 1].bool()
        return pycox_nll_pmf(preds, durations, events, self.reduction)


################################################################################
# NLLMTLRLoss
################################################################################

class NLLMTLRLoss(SurvivalLossBase):
    """
    Discrete-time survival loss for MTLR (Multi-Task Logistic Regression) (PyCox).

    forward(self, preds, targets):
      - preds: [N, T], real-valued
      - targets: shape [N, 2]
        durations = targets[:, 0]
        events    = targets[:, 1]
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        S = torch.sigmoid(scores).clamp(1e-9, 1 - 1e-9)
        S_prev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], dim=1)
        h = 1 - (S / S_prev).clamp(1e-9, 1 - 1e-9)
        return -torch.sum(torch.log1p(-h), dim=1)
    
    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0].long()
        events = targets[:, 1].bool()
        return pycox_nll_mtlr(preds, durations, events, self.reduction)


################################################################################
# BCESurvLoss
################################################################################

class BCESurvLoss(SurvivalLossBase):
    """
    Discrete-time survival loss using a series of binary classifiers for each time (PyCox).

    forward(self, preds, targets):
      - preds: [N, T], real-valued logits for survival
      - targets: shape [N, 2]
        durations = targets[:, 0]
        events    = targets[:, 1]
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        S = torch.sigmoid(scores).clamp(1e-9, 1 - 1e-9)
        S_prev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], dim=1)
        h = 1 - (S / S_prev).clamp(1e-9, 1 - 1e-9)
        return -torch.sum(torch.log1p(-h), dim=1)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0].long()
        events = targets[:, 1].bool()
        return pycox_bce_surv_loss(preds, durations, events, self.reduction)



################################################################################
# AdaptedCrossEntropySurvivalLoss
################################################################################

class AdaptedCrossEntropySurvivalLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        self.require_attention = False
        
    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        h = scores.clamp(1e-9, 1 - 1e-9)
        return -torch.sum(torch.log1p(-h), dim=1)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        N, T = preds.shape
        durations = targets[:, 0].long().clamp(0, T-1)   # 0-based
        events = targets[:, 1].bool()
        preds = preds.clamp(1e-9, 1 - 1e-9)              # hazards in [0,1]
        total = preds.new_tensor(0.)
        for i in range(N):
            t = int(durations[i].item())
            h = preds[i]
            if not events[i]:
                loss_i = -torch.log1p(-h[:t+1]).sum()                  # t inclusive
            else:
                pre = -torch.log1p(-h[:t]).sum() if t > 0 else h.new_tensor(0.)
                loss_i = pre + (-torch.log(h[t]))
            total += loss_i
        return total / N
