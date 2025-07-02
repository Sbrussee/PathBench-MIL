import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple

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
        return self.criterion(preds.float(), targets.float())


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
        ce_loss = self.cross_entropy_loss(preds, targets)
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
        ce_loss = self.cross_entropy(preds, targets)
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
        ce_loss = self.cross_entropy(preds, targets)
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
        ce_loss = self.cross_entropy(preds, targets)
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


################################################################################
# CoxPHLoss
################################################################################

class CoxPHLoss(nn.Module):
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

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0]
        events = targets[:, 1]
        log_h = preds.view(-1)  # same as old log_h
        durations = durations.view(-1)
        events = events.view(-1)
        # Use pycox
        return pycox_cox_ph_loss(log_h, durations, events)


################################################################################
# ExponentialConcordanceLoss
################################################################################

class ExponentialConcordanceLoss(nn.Module):
    """
    Exponential Concordance Loss (local/custom), continuous-time survival.

    forward(self, preds, targets):
      - preds: shape [N]
      - targets: shape [N, 2]
        durations = targets[:, 0]
        events    = targets[:, 1]
    """
    def __init__(self, event_weight=1.0, censored_weight=1.0):
        super().__init__()
        self.event_weight = event_weight
        self.censored_weight = censored_weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0]
        events = targets[:, 1]

        preds = preds.view(-1)
        n = len(preds)
        dur_i = durations.unsqueeze(0).repeat(n, 1)
        dur_j = durations.unsqueeze(1).repeat(1, n)
        evt_i = events.unsqueeze(0).repeat(n, 1)
        evt_j = events.unsqueeze(1).repeat(1, n)
        p_i = preds.unsqueeze(0).repeat(n, 1)
        p_j = preds.unsqueeze(1).repeat(1, n)

        valid_pairs = (dur_i < dur_j)
        event_i_mask = (evt_i == 1)
        final_mask = valid_pairs & event_i_mask

        w_i = torch.where(evt_i == 1, self.event_weight, self.censored_weight)
        w_j = torch.where(evt_j == 1, self.event_weight, self.censored_weight)
        pair_weights = w_i * w_j

        pred_diff = p_i - p_j
        exp_neg_diff = torch.exp(-pred_diff)
        loss_matrix = exp_neg_diff * final_mask.float() * pair_weights

        loss = loss_matrix.sum()
        num_pairs = final_mask.float().sum()
        if num_pairs > 0:
            loss = loss / num_pairs
        else:
            loss = torch.tensor(0.0, dtype=preds.dtype, device=preds.device)
        return loss


################################################################################
# RankingLoss
################################################################################

class RankingLoss(nn.Module):
    """
    Margin-based ranking loss (local/custom) for continuous-time survival.

    forward(self, preds, targets):
      - preds: shape [N]
      - targets: shape [N, 2]
        durations = targets[:, 0]
        events    = targets[:, 1]
      If durations[i]<durations[j] & event[i]=1 => penalize if preds[i] <= preds[j] by margin.
    """
    def __init__(self, margin=1.0, event_weight=1.0, censored_weight=1.0):
        super().__init__()
        self.margin = margin
        self.event_weight = event_weight
        self.censored_weight = censored_weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0]
        events = targets[:, 1]

        preds = preds.view(-1)
        n = len(durations)
        loss_val = 0.0
        num_pairs = 0
        for i in range(n):
            for j in range(n):
                if durations[i] < durations[j] and events[i] == 1:
                    weight = self.event_weight if events[i] == 1 else self.censored_weight
                    loss_val += weight * torch.relu(self.margin - (preds[i] - preds[j]))
                    num_pairs += 1
        if num_pairs > 0:
            loss_val /= num_pairs
        return loss_val


################################################################################
# NLLLogisticHazardLoss
################################################################################

class NLLLogisticHazardLoss(nn.Module):
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

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0].long()
        events = targets[:, 1]
        return pycox_nll_logistic_hazard(preds, durations, events, self.reduction)


################################################################################
# NLLPMFLoss
################################################################################

class NLLPMFLoss(nn.Module):
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

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0].long()
        events = targets[:, 1]
        return pycox_nll_pmf(preds, durations, events, self.reduction)


################################################################################
# NLLMTLRLoss
################################################################################

class NLLMTLRLoss(nn.Module):
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

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0].long()
        events = targets[:, 1]
        return pycox_nll_mtlr(preds, durations, events, self.reduction)


################################################################################
# BCESurvLoss
################################################################################

class BCESurvLoss(nn.Module):
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

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0].long()
        events = targets[:, 1]
        return pycox_bce_surv_loss(preds, durations, events, self.reduction)


################################################################################
# DeepHitLoss (local/custom simplified single-risk approach)
################################################################################

class DeepHitLoss(nn.Module):
    """
    Local/custom discrete survival 'DeepHit' style loss:
      - Negative log-likelihood at the observed/censored index
      - Ranking-based penalty

    forward(self, preds, targets):
      - preds: [N, T], typically probabilities or hazards in [0,1]
      - targets: [N, 2]
        durations = targets[:, 0]   (discrete time indices)
        events    = targets[:, 1]   (1=event, 0=censor)
    """
    def __init__(self, alpha=0.5, event_weight=1.0, censored_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.event_weight = event_weight
        self.censored_weight = censored_weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        B, T = preds.shape
        durations = targets[:, 0].long()
        events = targets[:, 1]

        # interpret preds as hazard or prob
        preds = preds.clamp(min=1e-12, max=1 - 1e-12)

        # NLL part
        dur_idx = (durations - 1).clamp(min=0, max=T - 1)
        idx = torch.arange(B, device=preds.device)
        likelihood_loss = -torch.log(preds[idx, dur_idx]) * events
        w = torch.where(events == 1, self.event_weight, self.censored_weight)
        likelihood_loss = likelihood_loss * w

        # Ranking part
        rank_loss = 0.0
        count = 0
        for i in range(B):
            for j in range(B):
                if durations[i] < durations[j] and events[i] == 1:
                    # event_weight only on the i-th subject
                    rank_loss += F.relu(preds[j, dur_idx[i]] - preds[i, dur_idx[i]]) * self.event_weight
                    count += 1
        if count > 0:
            rank_loss /= count

        total = self.alpha * likelihood_loss + (1 - self.alpha) * rank_loss
        return total.mean()


################################################################################
# AdaptedCrossEntropySurvivalLoss
################################################################################

class AdaptedCrossEntropySurvivalLoss(nn.Module):
    """
    Local/custom 'adapted cross-entropy' discrete survival from:
      Long et al. "Revisiting Cross-Entropy for Deep Survival Models..."

    forward(self, preds, targets):
      - preds: [N, T], hazard probabilities in [0,1]
      - targets: [N, 2]
        durations = targets[:, 0]  (in [1..T])
        events    = targets[:, 1]  (1=event, 0=censored)
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        self.require_attention = False

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        N, T = preds.shape
        durations = targets[:, 0]
        events = targets[:, 1]

        total_loss = 0.0
        for i in range(N):
            t_i = int(durations[i].item())
            c_i = int(events[i].item())

            # clamp t_i to [1, T]
            t_i = max(1, min(t_i, T))

            # hazards for subject i: clamp to avoid log(0)
            hazards_i = preds[i].clamp(min=self.eps, max=1.0 - self.eps)

            if c_i == 0:
                # censored => sum_{t=1..t_i} -log(1 - h(t))
                loss_i = 0.0
                for t in range(1, t_i + 1):
                    h_t = hazards_i[t - 1]
                    loss_i += -torch.log((1.0 - h_t).clamp(min=self.eps))
            else:
                # event => sum_{t=t_i..T} -log(h(t))
                loss_i = 0.0
                for t in range(t_i, T + 1):
                    h_t = hazards_i[t - 1]
                    loss_i += -torch.log(h_t.clamp(min=self.eps))

            total_loss += loss_i

        return total_loss / N
