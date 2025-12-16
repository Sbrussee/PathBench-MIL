import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple


### Helpers

def _register_opt_buffer(module: nn.Module, name: str, value: Optional[Tensor], *, dtype=torch.float32) -> None:
    """Register `value` as a (non-persistent) buffer if not None; else set attr to None."""
    if value is None:
        setattr(module, name, None)
    else:
        module.register_buffer(name, value.to(dtype=dtype), persistent=False)

def _like(buf: Optional[Tensor], like: Tensor) -> Optional[Tensor]:
    """Return `buf` cast to the same dtype as `like`. Device is already handled by register_buffer + .to()."""
    if buf is None:
        return None
    # device is already correct because buffers follow the module to GPU/CPU.
    return buf.to(dtype=like.dtype)

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
        _register_opt_buffer(self, 'weight', weight)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)
        ce_loss = F.cross_entropy(_float_logits(preds), targets, weight=_like(self.weight, preds), reduction='none')
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
        _register_opt_buffer(self, 'weight', weight)
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
            w = _like(self.weight, log_preds)
            true_dist *= w[targets].view(-1, 1)
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

    

################################################################################
# CoxPHLoss
################################################################################

class CoxPHLoss(nn.Module):
    """
    Weighted Cox partial log-likelihood.
    - If sample_weight is None: use pycox_cox_ph_loss (unchanged).
    - Else: weighted implementation (per-event weighting).
    """
    def __init__(self, event_weight=1.0, censored_weight=1.0,
                 sample_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.event_weight = event_weight
        self.censored_weight = censored_weight
        self.sample_weight = sample_weight

    @torch.no_grad()
    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        return scores.view(-1).float()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        durations = targets[:, 0].view(-1)
        events = targets[:, 1].view(-1).bool()
        eta = preds.view(-1)

        if self.sample_weight is None:
            # ✅ keep PyCox when unweighted
            return pycox_cox_ph_loss(eta, durations, events)

        # --- weighted Cox (per-event weights) ---
        order = torch.argsort(durations, descending=True)
        eta = eta[order]
        events = events[order]
        sw = self.sample_weight[order].float()

        # log sum exp risk set
        maxv = torch.max(eta)
        exp_eta = torch.exp(eta - maxv)
        cumsum_exp = torch.cumsum(exp_eta, dim=0)
        log_risk = torch.log(cumsum_exp) + maxv
        contrib = (eta - log_risk)[events]
        w_evt = sw[events]
        denom = w_evt.sum().clamp_min(1e-9)
        return -(contrib * w_evt).sum() / denom


################################################################################
# ExponentialConcordanceLoss
################################################################################

class ExponentialConcordanceLoss(nn.Module):
    """
    If sample_weight is None: same behavior.
    Else: weight event side by sample_weight.
    """
    def __init__(self, event_weight=1.0, censored_weight=1.0,
                 sample_weight: Optional[Tensor] = None):
        super().__init__()
        self.event_weight = event_weight
        self.censored_weight = censored_weight
        self.sample_weight = sample_weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        t = targets[:, 0]
        e = targets[:, 1].bool()
        s = preds.view(-1)

        finite = torch.isfinite(t) & torch.isfinite(s)
        if finite.sum() < 2:
            return s.sum() * 0.0

        t, e, s = t[finite], e[finite], s[finite]
        n = s.numel()
        dur_i = t.view(1, n).repeat(n, 1)
        dur_j = t.view(n, 1).repeat(1, n)
        evt_i = e.view(1, n).repeat(n, 1)
        mask = (dur_i < dur_j) & evt_i
        if not mask.any():
            return s.sum() * 0.0

        diff = s.view(1, n) - s.view(n, 1)  # p_i - p_j
        loss_mat = torch.exp(-diff) * mask.float()

        if self.sample_weight is not None:
            sw = self.sample_weight[finite].view(1, n).repeat(n, 1)
            loss_mat = loss_mat * sw

        denom = mask.float().sum().clamp_min(1.0)
        return loss_mat.sum() / denom

################################################################################
# RankingLoss
################################################################################

class RankingLoss(nn.Module):
    """
    If sample_weight is None: same behavior.
    Else: weight event side by sample_weight.
    """
    def __init__(self, margin=1.0, event_weight=1.0, censored_weight=1.0,
                 sample_weight: Optional[Tensor] = None):
        super().__init__()
        self.margin = margin
        self.event_weight = event_weight
        self.censored_weight = censored_weight
        self.sample_weight = sample_weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        t = targets[:, 0]
        e = targets[:, 1].bool()
        s = preds.view(-1)

        finite = torch.isfinite(t) & torch.isfinite(s)
        if finite.sum() < 2:
            return s.sum() * 0.0

        t, e, s = t[finite], e[finite], s[finite]
        n = s.numel()
        dur_i = t.view(n, 1)
        dur_j = t.view(1, n)
        evt_i = e.view(n, 1)
        mask = (dur_i < dur_j) & evt_i
        if not mask.any():
            return s.sum() * 0.0

        diff = s.view(n, 1) - s.view(1, n)
        loss_mat = torch.relu(self.margin - diff) * mask.float()

        if self.sample_weight is not None:
            sw = self.sample_weight[finite].view(n, 1)
            loss_mat = loss_mat * sw

        denom = mask.float().sum().clamp_min(1.0)
        return loss_mat.sum() / denom

################################################################################
# NLLLogisticHazardLoss
################################################################################

class NLLLogisticHazardLoss(nn.Module):
    """
    preds: [N, T] logits for hazards
    weight: Optional[T] per-bin weights
    sample_weight: Optional[N] per-sample weights
    If both are None -> use pycox_nll_logistic_hazard (unchanged).
    """
    def __init__(self, reduction: str = 'mean',
                 weight: Optional[Tensor] = None,
                 sample_weight: Optional[Tensor] = None):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.sample_weight = sample_weight

    @torch.no_grad()
    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        h = torch.sigmoid(scores)
        return -torch.sum(torch.log1p(-h), dim=1)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        N, T = preds.shape
        d = targets[:, 0].long().clamp(0, T - 1)
        e = targets[:, 1].bool()

        # ✅ pure PyCox path if no weights
        if self.weight is None and self.sample_weight is None:
            return pycox_nll_logistic_hazard(preds, d, e, self.reduction)

        # weighted path (same math, explicit terms)
        h = torch.sigmoid(preds).clamp(1e-9, 1 - 1e-9)
        neglog_1mh = -torch.log1p(-h)  # [N, T]
        neglog_h   = -torch.log(h)     # [N, T]

        if self.weight is not None:
            w = self.weight.to(h).view(1, -1)
            neglog_1mh = neglog_1mh * w
            neglog_h   = neglog_h * w

        per_sample = preds.new_zeros(N)
        for i in range(N):
            ti = int(d[i].item())
            if not e[i]:
                per_sample[i] = neglog_1mh[i, :ti+1].sum()
            else:
                pre = neglog_1mh[i, :ti].sum() if ti > 0 else per_sample.new_zeros(())
                per_sample[i] = pre + neglog_h[i, ti]

        if self.sample_weight is not None:
            sw = self.sample_weight.to(per_sample)
            return (per_sample * sw).sum() / sw.sum().clamp_min(1e-9)
        return per_sample.mean()


################################################################################
# NLLPMFLoss
################################################################################

class NLLPMFLoss(nn.Module):
    """
    preds: [N, T] logits -> softmax PMF
    weight: Optional[T] per-bin weights
    sample_weight: Optional[N] per-sample weights
    """
    def __init__(self, reduction: str = 'mean',
                 weight: Optional[Tensor] = None,
                 sample_weight: Optional[Tensor] = None):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.sample_weight = sample_weight

    @torch.no_grad()
    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        pmf = torch.softmax(scores, dim=-1).clamp(1e-9, 1-1e-9)
        N, K = pmf.shape
        S_prev = torch.ones(N, device=scores.device, dtype=scores.dtype)
        H = torch.zeros(N, device=scores.device, dtype=scores.dtype)
        for k in range(K):
            h_k = (pmf[:, k] / S_prev).clamp(1e-9, 1 - 1e-9)
            H += -torch.log1p(-h_k)
            S_prev *= (1 - h_k)
        return H

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        N, T = preds.shape
        d = targets[:, 0].long().clamp(0, T - 1)
        e = targets[:, 1].bool()

        if self.weight is None and self.sample_weight is None:
            return pycox_nll_pmf(preds, d, e, self.reduction)

        pmf = torch.softmax(preds, dim=-1).clamp(1e-9, 1-1e-9)
        cdf = torch.cumsum(pmf, dim=1)
        per_sample = preds.new_zeros(N)

        w = None
        if self.weight is not None:
            w = self.weight.to(pmf.device, pmf.dtype)

        for i in range(N):
            ti = int(d[i].item())
            if e[i]:
                ell = -torch.log(pmf[i, ti])
                if w is not None: ell = ell * w[ti]
            else:
                Sd = (1.0 - cdf[i, ti]).clamp(1e-9, 1.0)
                ell = -torch.log(Sd)
                if w is not None:
                    ell = ell * (w[:ti+1].mean())
            per_sample[i] = ell

        if self.sample_weight is not None:
            sw = self.sample_weight.to(per_sample)
            return (per_sample * sw).sum() / sw.sum().clamp_min(1e-9)
        return per_sample.mean()


################################################################################
# NLLMTLRLoss
################################################################################

class NLLMTLRLoss(nn.Module):
    """
    preds: [N, T] raw MTLR scores.
    weight: Optional[T] per-bin weights
    sample_weight: Optional[N] per-sample weights
    """
    def __init__(self, reduction: str = 'mean',
                 weight: Optional[Tensor] = None,
                 sample_weight: Optional[Tensor] = None):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.sample_weight = sample_weight

    @torch.no_grad()
    def risk(self, scores: Tensor) -> Tensor:
        rev_cumsum = torch.flip(torch.cumsum(torch.flip(scores, dims=[1]), dim=1), dims=[1])
        pmf = torch.softmax(rev_cumsum, dim=1)
        eps = 1e-9
        N, K = pmf.shape
        S_prev = torch.ones(N, device=scores.device, dtype=scores.dtype)
        H = torch.zeros(N, device=scores.device, dtype=scores.dtype)
        for k in range(K):
            h_k = (pmf[:, k] / S_prev).clamp(eps, 1 - eps)
            H += -torch.log1p(-h_k)
            S_prev *= (1 - h_k)
        return H

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        N, T = preds.shape
        d = targets[:, 0].long().clamp(0, T - 1)
        e = targets[:, 1].bool()

        if self.weight is None and self.sample_weight is None:
            return pycox_nll_mtlr(preds, d, e, self.reduction)

        tails = torch.flip(torch.cumsum(torch.flip(preds, dims=[1]), dim=1), dims=[1])
        log_p = tails - torch.logsumexp(tails, dim=1, keepdim=True)
        cdf_log = torch.logcumsumexp(torch.exp(log_p), dim=1)

        per_sample = preds.new_zeros(N)
        w = None
        if self.weight is not None:
            w = self.weight.to(preds.device, preds.dtype)

        for i in range(N):
            ti = int(d[i].item())
            if e[i]:
                ell = -log_p[i, ti]
                if w is not None: ell = ell * w[ti]
            else:
                log_Sd = torch.log1p(-torch.exp(cdf_log[i, ti]).clamp(0, 1-1e-9))
                ell = -log_Sd
                if w is not None:
                    ell = ell * (w[:ti+1].mean())
            per_sample[i] = ell

        if self.sample_weight is not None:
            sw = self.sample_weight.to(per_sample)
            return (per_sample * sw).sum() / sw.sum().clamp_min(1e-9)
        return per_sample.mean()


################################################################################
# BCESurvLoss
################################################################################

class BCESurvLoss(nn.Module):
    """
    preds: [N, T] logits for survival S_k
    weight: Optional[T] per-bin weights for BCE terms
    sample_weight: Optional[N] per-sample weights
    """
    def __init__(self, reduction: str = 'mean',
                 weight: Optional[Tensor] = None,
                 sample_weight: Optional[Tensor] = None):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.sample_weight = sample_weight

    @torch.no_grad()
    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        S = torch.sigmoid(scores).clamp(1e-9, 1-1e-9)
        S_prev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], dim=1)
        h = 1 - (S / S_prev).clamp(1e-9, 1 - 1e-9)
        return -torch.sum(torch.log1p(-h), dim=1)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        N, T = preds.shape
        d = targets[:, 0].long().clamp(0, T - 1)
        e = targets[:, 1].bool()

        if self.weight is None and self.sample_weight is None:
            # keep original PyCox when unweighted
            return pycox_bce_surv_loss(preds, d, e, self.reduction)

        S = torch.sigmoid(preds).clamp(1e-9, 1-1e-9)
        y = torch.ones_like(S)
        mask = torch.ones_like(S)

        for i in range(N):
            ti = int(d[i].item())
            if e[i]:
                y[i, ti] = 0.0
                y[i, ti+1:] = 0.0
            else:
                y[i, ti+1:] = 0.0
                mask[i, ti+1:] = 0.0

        bce = F.binary_cross_entropy(S, y, reduction='none') * mask

        if self.weight is not None:
            w = self.weight.to(S.dtype).to(S.device).view(1, -1)
            bce = bce * w

        per_sample = bce.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        if self.sample_weight is not None:
            sw = self.sample_weight.to(per_sample)
            return (per_sample * sw).sum() / sw.sum().clamp_min(1e-9)
        return per_sample.mean()



################################################################################
# AdaptedCrossEntropySurvivalLoss
################################################################################

class AdaptedCrossEntropySurvivalLoss(nn.Module):
    """
    preds: [N, T] hazards in [0,1]
    weight: Optional[T] per-bin weights
    sample_weight: Optional[N] per-sample weights
    """
    def __init__(self, eps: float = 1e-7,
                 weight: Optional[Tensor] = None,
                 sample_weight: Optional[Tensor] = None):
        super().__init__()
        self.eps = eps
        self.weight = weight
        self.sample_weight = sample_weight
        self.require_attention = False

    @torch.no_grad()
    def risk(self, scores: torch.Tensor) -> torch.Tensor:
        h = scores.clamp(1e-9, 1 - 1e-9)
        return -torch.sum(torch.log1p(-h), dim=1)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        N, T = preds.shape
        d = targets[:, 0].long().clamp(0, T - 1)
        e = targets[:, 1].bool()
        h = preds.clamp(1e-9, 1 - 1e-9)

        neglog_1mh = -torch.log1p(-h)  # [N, T]
        neglog_h   = -torch.log(h)     # [N, T]

        if self.weight is not None:
            w = self.weight.to(h).view(1, -1)
            neglog_1mh = neglog_1mh * w
            neglog_h   = neglog_h * w

        per_sample = preds.new_zeros(N)
        for i in range(N):
            ti = int(d[i].item())
            if not e[i]:
                per_sample[i] = neglog_1mh[i, :ti+1].sum()
            else:
                pre = neglog_1mh[i, :ti].sum() if ti > 0 else per_sample.new_zeros(())
                per_sample[i] = pre + neglog_h[i, ti]

        if self.sample_weight is not None:
            sw = self.sample_weight.to(per_sample)
            return (per_sample * sw).sum() / sw.sum().clamp_min(1e-9)
        return per_sample.mean()