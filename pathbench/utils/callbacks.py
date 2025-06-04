from fastai.callback.core import Callback

class ReduceLROnPlateau(Callback):
    "A simple ReduceLROnPlateau callback that adjusts LR when a monitored metric plateaus."
    order = 60  # Ensure this runs after most other callbacks
    def __init__(self, monitor='valid_loss', factor=0.1, patience=2, min_lr=1e-7, verbose=False):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

    def before_fit(self):
        self.best = float('inf')
        self.num_bad_epochs = 0

    def after_epoch(self):
        # Assume that valid_loss is the first metric in recorder.values.
        # If you have a different ordering, adjust the index accordingly.
        if not self.recorder.values: return
        current = self.recorder.values[-1][0]
        if current < self.best:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for i, pg in enumerate(self.opt.param_groups):
                new_lr = max(pg['lr'] * self.factor, self.min_lr)
                if self.verbose:
                    print(f"Reducing lr for group {i} from {pg['lr']:.2e} to {new_lr:.2e}")
                pg['lr'] = new_lr
            self.num_bad_epochs = 0  # Reset the counter after a reduction