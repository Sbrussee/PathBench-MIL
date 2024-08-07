from fastai.learner import Metric
from lifelines.utils import concordance_index
import torch

class ConcordanceIndex(Metric):
    """Concordance index metric for survival analysis."""
    def __init__(self):
        self.name = "concordance_index"
        self.reset()

    def reset(self):
        """Reset the metric."""
        self.preds, self.durations, self.events = [], [], []

    def accumulate(self, learn):
        """Accumulate predictions and targets from a batch."""
        preds = learn.pred
        targets = learn.y
        self.accum_values(preds, targets)

    def accum_values(self, preds, targets):
        """Accumulate predictions and targets from a batch."""
        preds, targets = to_detach(preds), to_detach(targets)

        # Ensure preds are tensors, handle dict, tuple, and list cases
        if isinstance(preds, dict):
            preds = torch.cat([torch.tensor(v).view(-1) if not isinstance(v, torch.Tensor) else v.view(-1) for v in preds.values()])
        elif isinstance(preds, tuple):
            preds = torch.cat([torch.tensor(p).view(-1) if not isinstance(p, torch.Tensor) else p.view(-1) for p in preds])
        elif isinstance(preds, list):
            preds = torch.cat([torch.tensor(p).view(-1) if not isinstance(p, torch.Tensor) else p.view(-1) for p in preds])
        else:
            preds = preds.view(-1) if isinstance(preds, torch.Tensor) else torch.tensor(preds).view(-1)

        # Handle survival targets (durations and events)
        durations = targets[:, 0].view(-1)
        events = targets[:, 1].view(-1)
        
        self.preds.append(preds)
        self.durations.append(durations)
        self.events.append(events)

    @property
    def value(self):
        """Calculate the concordance index."""
        if len(self.preds) == 0: return None
        preds = torch.cat(self.preds).cpu().numpy()
        durations = torch.cat(self.durations).cpu().numpy()
        events = torch.cat(self.events).cpu().numpy()
        ci = concordance_index(durations, preds, events)
        return ci

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value