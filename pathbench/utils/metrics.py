from fastai.learner import Metric
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
import torch
from fastai.torch_core import to_detach, flatten_check
import logging

class ConcordanceIndex(Metric):
    """
    Concordance index metric for survival analysis, supporting both:
      (1) Continuous survival predictions: preds.shape == (batch_size,)
      (2) Discrete survival predictions:   preds.shape == (batch_size, n_bins)
          We automatically convert discrete to continuous by a weighted sum
          over the bin indices [1..n_bins].
    """

    def __init__(self):
        self._name = "concordance_index"
        self.reset()

    def reset(self):
        """Reset the metric."""
        self.preds = []
        self.durations = []
        self.events = []

    def accumulate(self, learn):
        """Accumulate predictions and targets from a batch."""
        preds = learn.pred
        targets = learn.y
        self.accum_values(preds, targets)

    def accum_values(self, preds, targets):
        """
        Accumulate predictions and targets from a batch.

        - targets must be shape (batch_size, 2) -> [duration, event].
        - preds can be (batch_size,) for continuous survival
          or (batch_size, n_bins) for discrete survival.

        We automatically interpret 2D preds as discrete probabilities
        and convert them to a single predicted time by a weighted sum
        with bin indices [1..n_bins].
        """
        preds, targets = to_detach(preds), to_detach(targets)

        # Flatten or convert any containers into a single tensor
        preds = self._flatten_preds(preds)

        # durations = targets[:, 0], events = targets[:, 1]
        durations = targets[:, 0].float()
        events = targets[:, 1].float()

        # If we have discrete predictions, convert to continuous
        if preds.dim() > 1 and preds.shape[1] > 1:
            #Check if preds are probabilities
            if not torch.all(preds.sum(dim=1) == 1.0):
                preds = torch.softmax(preds, dim=-1)
            #Perform weighted sum over bins
            preds = (preds * torch.arange(1, preds.shape[1] + 1).float()).sum(dim=1)


        #Check if preds should be 

        # Accumulate for final c-index computation
        self.preds.append(preds)
        self.durations.append(durations)
        self.events.append(events)

    def _flatten_preds(self, preds):
        """
        Flattens nested structures of predictions, if needed.
        E.g. if preds is a tuple/list, or has multiple dimensions.
        """
        # For your use case, you might just do: return preds.view(-1, *preds.shape[2:])
        # But here's a robust fallback:
        if isinstance(preds, (list, tuple)):
            # Attempt to concatenate or flatten
            preds = torch.cat([p.view(-1, *p.shape[2:]) if p.dim() > 1 else p.flatten()
                               for p in preds], dim=0)
        return preds

    @property
    def value(self):
        """Calculate the concordance index using lifelines."""
        if len(self.preds) == 0:
            return None

        # Concatenate all predictions/durations/events across mini-batches
        preds = torch.cat(self.preds).cpu().numpy()
        durations = torch.cat(self.durations).cpu().numpy()
        events = torch.cat(self.events).cpu().numpy()

        # events must be bool or 0/1 for lifelines
        ci = concordance_index(durations, preds, events)
        return ci

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