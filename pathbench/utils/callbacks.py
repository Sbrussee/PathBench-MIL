from fastai.callback.core import Callback, CancelFitException

class ReduceLROnPlateau(Callback):
    "Reduce LR by `factor` when `monitor` stops improving for `patience` epochs."
    order = 60  # after most callbacks
    def __init__(self, monitor='valid_loss', factor=0.1, patience=2, min_lr=1e-7, verbose=False):
        self.monitor, self.factor, self.patience, self.min_lr, self.verbose = \
            monitor, factor, patience, min_lr, verbose

    def before_fit(self):
        # find which column in recorder.values we should look at
        self.metric_names = self.recorder.metric_names  # e.g. ['train_loss','valid_loss',…]
        try:
            self.idx = self.metric_names.index(self.monitor)
        except ValueError:
            raise ValueError(f"Monitor ‘{self.monitor}’ not found in {self.metric_names}")
        self.best = float('inf')
        self.num_bad_epochs = 0

    def after_epoch(self):
        vals = self.recorder.values
        # if no values yet, skip
        if not vals: 
            return
        # grab the monitored metric from the last epoch
        current = vals[-1][self.idx]
        if current < self.best:
            self.best, self.num_bad_epochs = current, 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for i, pg in enumerate(self.opt.param_groups):
                new_lr = max(pg['lr'] * self.factor, self.min_lr)
                if self.verbose:
                    print(f"Reducing LR group {i}: {pg['lr']:.2e} → {new_lr:.2e}")
                pg['lr'] = new_lr
            self.num_bad_epochs = 0  # reset


class EarlyStoppingCallback(Callback):
    "Stop training when `monitor` stops improving for `patience` epochs."
    order = 70  # after ReduceLROnPlateau
    def __init__(self, monitor='valid_loss', patience=5, verbose=False):
        self.monitor, self.patience, self.verbose = monitor, patience, verbose

    def before_fit(self):
        # same indexing logic as above
        self.metric_names = self.recorder.metric_names
        try:
            self.idx = self.metric_names.index(self.monitor)
        except ValueError:
            raise ValueError(f"Monitor ‘{self.monitor}’ not found in {self.metric_names}")
        self.best = float('inf')
        self.num_bad_epochs = 0

    def after_epoch(self):
        vals = self.recorder.values
        if not vals:
            return
        current = vals[-1][self.idx]
        if current < self.best:
            self.best, self.num_bad_epochs = current, 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            if self.verbose:
                print(f"Early stopping after {self.num_bad_epochs} bad epochs.")
            # Use the FastAI‐provided exception so fit() knows to clean up
            raise CancelFitException()