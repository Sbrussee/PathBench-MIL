Extending PathBench
===================

PathBench is designed to be easily extensible. Here are some ways you can extend PathBench:

## Custom Feature Extractors

New feature extractors can be added to `pathbench/models/feature_extractors.py`. Follow this format:

```python
@register_torch
class kaiko_s8(TorchFeatureExtractor):
    tag = 'kaiko_s8'

    def __init__(self, tile_px=256):
        super().__init__()
        self.model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", 'vits8', trust_repo=True)
        self.model.to('cuda')
        self.num_features = 384
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.model.eval()
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {'class': 'kaiko_s8', 'kwargs': {}}
```

## Custom MIL Aggregators
New MIL aggregation models can be added to `pathbench/models/aggregators.py`. Follow this format:
```python
class lse_mil(nn.Module):
    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, r: float = 1.0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self.r = r

    def forward(self, bags):
        embeddings = self.encoder(bags)
        lse_pooling = self.r * torch.logsumexp(embeddings / self.r, dim=1)
        scores = self.head(lse_pooling)
        return scores
```

## Custom Losses
New loss functions can be added to `pathbench/utils/losses.py`. Follow this format:
```python
class CoxPHLoss(nn.Module):
    def forward(self, preds, targets):
        durations = targets[:, 0]
        events = targets[:, 1]
        if torch.sum(events) == 0:
            return torch.tensor(1e-6, dtype=preds.dtype, device=preds.device)
        loss = cox_ph_loss(preds, durations, events).float()
        return loss
```

## Custom Metrics
New metrics can be added to `pathbench/utils/metrics.py`. Follow this format:
```python
class ConcordanceIndex(Metric):
    def __init__(self):
        self.name = "concordance_index"
        self.reset()

    def reset(self):
        self.preds, self.durations, self.events = [], [], []

    def accumulate(self, learn):
        preds = learn.pred
        targets = learn.y
        self.accum_values(preds, targets)

    def accum_values(self, preds, targets):
        preds, targets = to_detach(preds), to_detach(targets)
        if isinstance(preds, dict):
            preds = torch.cat([torch.tensor(v).view(-1) if not isinstance(v, torch.Tensor) else v.view(-1) for v in preds.values()])
        elif isinstance(preds, tuple):
            preds = torch.cat([torch.tensor(p).view(-1) if not isinstance(p, torch.Tensor) else p.view(-1) for p in preds])
        elif isinstance(preds, list):
            preds = torch.cat([torch.tensor(p).view(-1) if not isinstance(p, torch.Tensor) else p.view(-1) for p in preds])
        else:
            preds = preds.view(-1) if isinstance(preds, torch.Tensor) else torch.tensor(preds).view(-1)

        durations = targets[:, 0].view(-1)
        events = targets[:, 1].view(-1)
        self.preds.append(preds)
        self.durations.append(durations)
        self.events.append(events)

    @property
    def value(self):
        if len(self.preds) == 0: return None
        preds = torch.cat(self.preds).cpu().numpy()
        durations = torch.cat(self.durations).cpu().numpy()
        events = torch.cat(self.events).cpu().numpy()
        ci = concordance_index(durations, preds, events)
        return ci
```

## Further Extension

- To add new visualization options, one can add them to pathbench/visualization/visualization.py and call them appropriately in pathbench/benchmarking/benchmark.py.
- To add new normalization methods, one needs to change code in the forked slideflow repository.