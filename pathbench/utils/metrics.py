from fastai.learner import Metric
from fastai.torch_core import to_detach
import torch
import numpy as np
from sksurv.metrics import concordance_index_censored

# ---- helpers ----
def _scores_to_hazards(preds_raw: torch.Tensor, link: str = "logit") -> torch.Tensor:
    """
    Map unbounded per-bin scores -> hazard probabilities in (0,1).
    - "logit":    score = logit(h)       -> h = sigmoid(score)
    - "cloglog":  score = η (cloglog)    -> h = 1 - exp(-exp(score))
    - "prob":     score = h              -> h = score
    - "logprob":  score = log(h)         -> h = exp(score)
    """
    x = preds_raw.float()
    if link == "logit":
        h = torch.sigmoid(x)
    elif link == "cloglog":
        h = 1.0 - torch.exp(-torch.exp(x))
    elif link == "prob":
        h = x
    elif link == "logprob":
        h = torch.exp(x)
    else:
        raise ValueError(f"Unknown link: {link}")
    eps = 1e-9
    return torch.clamp(h, eps, 1.0 - eps)

def _cumulative_hazard_risk(hazards: torch.Tensor) -> torch.Tensor:
    """Discrete-time risk proxy: risk = -sum(log(1 - h_k))  (higher = higher hazard)."""
    return -torch.sum(torch.log(1.0 - hazards), dim=1)

# ---- metric ----
class ConcordanceIndex(Metric):
    """
    Concordance index via sksurv for survival (continuous CoxPH or discrete-time).

    - Continuous (CoxPH): preds shape (N,) -> use as risk (higher=worse). Set invert_preds=True
      ONLY if your scores mean higher=better survival (rare for CoxPH).
    - Discrete: preds shape (N,K) -> convert scores to hazards (link), then
      risk = -∑ log(1 - h_k). Pass risk directly to sksurv.
    """
    def __init__(self, invert_preds: bool = False, link: str = "logit"):
        """
        invert_preds: set True only if larger model scores mean *better* survival.
        link: "logit" (NLLLogisticHazard), "cloglog", or "prob" (already probabilities).
        """
        self._name = "concordance_index"
        self.invert_preds = invert_preds
        self.link = link
        self.reset()

    def reset(self):
        self._risk = []
        self._durations = []
        self._events = []

    def accumulate(self, learn):
        self.accum_values(learn.pred, learn.y)

    def accum_values(self, preds, targets):
        preds = to_detach(preds)
        targets = to_detach(targets)

        durations = targets[:, 0].float()
        events = targets[:, 1].float()

        # Build a single risk vector (higher = worse)
        if preds.dim() == 1 or preds.shape[-1] == 1:
            # Continuous CoxPH-style: raw score is the linear predictor η
            risk = preds.view(-1).float()
        else:
            # Discrete-time: per-bin scores -> hazards -> cumulative hazard risk
            hazards = _scores_to_hazards(preds, link=self.link)
            risk = _cumulative_hazard_risk(hazards)

        if self.invert_preds:
            risk = -risk

        self._risk.append(risk)
        self._durations.append(durations)
        self._events.append(events)

    @property
    def value(self):
        if not self._risk:
            return None
        risk = torch.cat(self._risk).cpu().numpy().astype(np.float64)
        durations = torch.cat(self._durations).cpu().numpy().astype(np.float64)
        events = torch.cat(self._events).cpu().numpy().astype(np.int32).astype(bool)

        # sksurv: returns (c_index, concordant, discordant, tied_risk, tied_time)
        c_idx = float(concordance_index_censored(events, durations, risk)[0])
        return c_idx

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value



class safe_roc_auc(Metric):
    """ROC AUC metric for multiclass classification that
    can handle cases where only one class is present."""
    def __init__(self, average='macro', multi_class='ovr'):
        self.average = average
        self.multi_class = multi_class
        self.name = "safe_roc_auc"
        self.reset()

    def reset(self):
        """Reset the metric."""
        self.preds, self.targets = [], []

    def accumulate(self, learn):
        """Accumulate predictions and targets from a batch."""
        preds = learn.pred
        targets = learn.y
        self.accum_values(preds, targets)

    def accum_values(self, preds, targets):
        """Accumulate predictions and targets from a batch."""
        preds, targets = to_detach(preds), to_detach(targets)

        # Convert predictions to probabilities if they are logits
        if preds.shape[-1] > 1:
            preds = torch.softmax(preds, dim=-1)
        
        # Flatten predictions and targets if necessary
        preds = preds.view(-1, preds.shape[-1])
        targets = targets.view(-1)

        self.preds.append(preds)
        self.targets.append(targets)

    @property
    def value(self):
        """Calculate the ROC AUC score."""
        if len(self.preds) == 0: 
            return None
        
        preds = torch.cat(self.preds).cpu().numpy()
        targets = torch.cat(self.targets).cpu().numpy()

        # Handle case where only one class is present
        if len(set(targets)) < 2:
            return None
        
        try:
            auc = roc_auc_score(targets, preds, average=self.average, multi_class=self.multi_class)
        except ValueError as e:
            print(f"Error calculating ROC AUC: {e}")
            auc = None
        return auc

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value