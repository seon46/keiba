"""Neural strength model (PyTorch) for the v3 rank distribution: listwise Plackett-Luce over whole races.

Per runner: robust-scaled numeric features (+ missing flags), embeddings of the
categorical codes, the jockey and the trainer, and a pedigree vector = a
learned, path-weighted sum of the 62 five-generation ancestor embeddings
(shared table, so the same stallion as sire or as dam's sire shares one vector).
A small Transformer encoder then lets the runners of a race attend to each
other (pace, field strength) before a linear head gives the strength score.

Training uses every runner as context but the PL likelihood only over the
finishers (as the LightGBM model does), early-stopped on the PL loss of the
stop period.  The scores are on an arbitrary scale; stage temperatures are
fitted afterwards exactly as for LightGBM.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import time

ANCESTORS = 62


@dataclass
class NeuralConfig:
    width: int = 192
    embedding: int = 16
    pedigree_dim: int = 32
    layers: int = 2
    heads: int = 4
    dropout: float = .15
    ancestor_dropout: float = .1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_races: int = 256
    max_epochs: int = 40
    patience: int = 4
    ancestor_min_count: int = 2
    entity_min_count: int = 20
    seed: int = 42


def device_name(prefer='auto'):
    import torch
    if prefer == 'cpu':
        return 'cpu'
    if prefer == 'cuda' and not torch.cuda.is_available():
        raise ValueError('GPU（CUDA）を使えません。PyTorchのGPU版を入れてください（scripts\\setup-nn.ps1）。')
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

class Inputs:
    """Encoded arrays for every row of a group (fitted vocabularies / scaling come from ``fit_rows``)."""

    def __init__(self, matrix, categorical, race_ids, finished, positions, jockeys, trainers, pedigrees, horses,
                 fit_rows, config: NeuralConfig):
        import warnings
        import numpy as np
        warnings.filterwarnings('ignore', message='All-NaN slice', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='Degrees of freedom', category=RuntimeWarning)
        count, width = matrix.shape
        self.categorical = list(categorical)
        numeric = [column for column in range(width) if column not in set(self.categorical)]
        fit_values = matrix[np.ix_(fit_rows, numeric)].astype(np.float64)
        low = np.nanpercentile(fit_values, .5, axis=0)
        high = np.nanpercentile(fit_values, 99.5, axis=0)
        low, high = np.nan_to_num(low), np.nan_to_num(high)
        clipped = np.clip(fit_values, low, high)
        mean, spread = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0)
        mean, spread = np.nan_to_num(mean), np.where(np.nan_to_num(spread) > 1e-9, np.nan_to_num(spread), 1.0)
        values = matrix[:, numeric].astype(np.float64)
        missing = np.isnan(values)
        flagged = np.flatnonzero(missing[fit_rows].mean(axis=0) > .001)
        scaled = (np.clip(values, low, high) - mean) / spread
        scaled[missing] = 0.0
        self.numeric = np.concatenate([scaled, missing[:, flagged].astype(np.float64)], axis=1).astype(np.float32)
        self.scaling = dict(numeric=numeric, low=low, high=high, mean=mean, spread=spread, flagged=flagged)
        # categorical codes -> indices (0 = unseen / missing)
        self.cat_vocab = []
        codes = np.zeros((count, len(self.categorical)), dtype=np.int64)
        for position, column in enumerate(self.categorical):
            seen = sorted({float(value) for value in matrix[fit_rows, column] if np.isfinite(value)})
            vocab = {value: index + 1 for index, value in enumerate(seen)}
            self.cat_vocab.append(vocab)
            codes[:, position] = [vocab.get(float(value), 0) if np.isfinite(value) else 0 for value in matrix[:, column]]
        self.cat_codes = codes

        def vocabulary(values, minimum):
            counts = Counter(value for value in values if value)
            return {value: index + 1 for index, value in enumerate(sorted(key for key, n in counts.items() if n >= minimum))}

        self.jockey_vocab = vocabulary(jockeys[fit_rows].tolist(), config.entity_min_count)
        self.trainer_vocab = vocabulary(trainers[fit_rows].tolist(), config.entity_min_count)
        self.jockeys = np.asarray([self.jockey_vocab.get(value, 0) for value in jockeys.tolist()], dtype=np.int64)
        self.trainers = np.asarray([self.trainer_vocab.get(value, 0) for value in trainers.tolist()], dtype=np.int64)
        fit_horses = set(horses[fit_rows].tolist())
        ancestor_counts = Counter(key for horse in fit_horses for key in pedigrees.get(horse, ()) if key)
        self.ancestor_vocab = {key: index + 1 for index, key in enumerate(
            sorted(key for key, n in ancestor_counts.items() if n >= config.ancestor_min_count))}
        ancestry = np.zeros((count, ANCESTORS), dtype=np.int64)
        cache = {}
        for row, horse in enumerate(horses.tolist()):
            encoded = cache.get(horse)
            if encoded is None:
                keys = pedigrees.get(horse)
                encoded = cache[horse] = ([self.ancestor_vocab.get(key, 0) if key else 0 for key in keys]
                                          if keys else [0] * ANCESTORS)
            ancestry[row] = encoded
        self.ancestry = ancestry
        # races (rows contiguous per race)
        boundaries = np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1], True])
        self.starts, self.ends = boundaries[:-1], boundaries[1:]
        self.race_of_row = np.repeat(np.arange(len(self.starts)), self.ends - self.starts)
        self.finished, self.positions = finished.astype(bool), positions.astype(np.int64)
        self.cat_cards = [len(vocab) for vocab in self.cat_vocab]

    def races_for(self, rows, *, need_finishers=True):
        """Race indices whose rows are all in ``rows`` (and with >= 2 finishers when requested)."""
        import numpy as np
        selected = np.zeros(len(self.starts), dtype=bool)
        inside = np.zeros(len(self.race_of_row), dtype=bool)
        inside[rows] = True
        for race, (start, end) in enumerate(zip(self.starts, self.ends)):
            if inside[start:end].all():
                selected[race] = not need_finishers or self.finished[start:end].sum() >= 2
        return np.flatnonzero(selected)


def _batch(inputs, races, device):
    import numpy as np
    import torch
    sizes = inputs.ends[races] - inputs.starts[races]
    width = int(sizes.max())
    index = np.full((len(races), width), -1, dtype=np.int64)
    for slot, race in enumerate(races):
        start, end = inputs.starts[race], inputs.ends[race]
        index[slot, :end - start] = np.arange(start, end)
    mask = index >= 0
    safe = np.where(mask, index, 0)
    tensor = lambda array, dtype=None: torch.as_tensor(array, device=device) if dtype is None else torch.as_tensor(array, dtype=dtype, device=device)
    return dict(index=index, mask=tensor(mask), numeric=tensor(inputs.numeric[safe]), cat=tensor(inputs.cat_codes[safe]),
                jockey=tensor(inputs.jockeys[safe]), trainer=tensor(inputs.trainers[safe]),
                ancestry=tensor(inputs.ancestry[safe]),
                finished=tensor(inputs.finished[safe] & mask), positions=tensor(inputs.positions[safe]))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_model(inputs, config: NeuralConfig):
    import torch
    from torch import nn

    class RaceNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cat_embeddings = nn.ModuleList(nn.Embedding(card + 1, min(8, card + 1)) for card in inputs.cat_cards)
            self.jockey = nn.Embedding(len(inputs.jockey_vocab) + 1, config.embedding, padding_idx=0)
            self.trainer = nn.Embedding(len(inputs.trainer_vocab) + 1, config.embedding, padding_idx=0)
            self.ancestor = nn.Embedding(len(inputs.ancestor_vocab) + 1, config.pedigree_dim, padding_idx=0)
            generations = torch.tensor([len(path) for path in _paths()], dtype=torch.float32)
            self.path_weight = nn.Parameter(.5 ** generations)
            width = (inputs.numeric.shape[1] + sum(min(8, card + 1) for card in inputs.cat_cards)
                     + 2 * config.embedding + config.pedigree_dim)
            self.encoder = nn.Sequential(nn.Linear(width, config.width), nn.GELU(), nn.Dropout(config.dropout),
                                         nn.Linear(config.width, config.width), nn.GELU())
            layer = nn.TransformerEncoderLayer(config.width, config.heads, 2 * config.width, config.dropout,
                                               batch_first=True, norm_first=True, activation='gelu')
            self.context = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, 1))

        def forward(self, batch):
            ancestry = batch['ancestry']
            if self.training and config.ancestor_dropout:
                ancestry = ancestry * (torch.rand(ancestry.shape, device=ancestry.device) >= config.ancestor_dropout)
            pedigree = (self.ancestor(ancestry) * self.path_weight[None, None, :, None]).sum(2)
            parts = [batch['numeric'], pedigree, self.jockey(batch['jockey']), self.trainer(batch['trainer'])]
            parts += [embedding(batch['cat'][..., position]) for position, embedding in enumerate(self.cat_embeddings)]
            hidden = self.encoder(torch.cat(parts, dim=-1))
            hidden = self.context(hidden, src_key_padding_mask=~batch['mask'])
            return self.head(hidden).squeeze(-1)

    return RaceNet()


def _paths():
    from .pedigree import PATHS
    return PATHS


def pl_loss(scores, batch):
    """Mean over races of the mean PL stage loss of the finishers (non-finishers excluded)."""
    import torch
    scores = scores.float()
    finished = batch['finished']
    big = torch.iinfo(torch.int64).max // 4
    key = torch.where(finished, batch['positions'], torch.full_like(batch['positions'], big))
    order = torch.argsort(key, dim=1)
    ordered = torch.gather(scores, 1, order)
    valid = torch.gather(finished, 1, order)
    masked = torch.where(valid, ordered, torch.full_like(ordered, -1e9))
    tail = torch.logcumsumexp(masked.flip(1), dim=1).flip(1)
    counts = valid.sum(1)
    stages = torch.arange(ordered.shape[1], device=scores.device)[None, :] < (counts - 1)[:, None]
    terms = torch.where(stages, tail - ordered, torch.zeros_like(ordered))
    per_race = terms.sum(1) / (counts - 1).clamp(min=1)
    return per_race.mean()


# --------------------------------------------------------------------------
# Training / prediction
# --------------------------------------------------------------------------

def train(inputs, fit_rows, stop_rows, config: NeuralConfig, *, device='auto', report=None):
    import numpy as np
    import torch
    report = report or (lambda **message: None)
    device = device_name(device)
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)
    fit_races, stop_races = inputs.races_for(fit_rows), inputs.races_for(stop_rows)
    if not len(fit_races) or not len(stop_races):
        raise ValueError('ニューラルネットの学習または早期終了用のレースが不足しています。')
    model = build_model(inputs, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps = math.ceil(len(fit_races) / config.batch_races) * config.max_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=steps,
                                                    pct_start=.1)
    use_amp = device == 'cuda'
    best, best_state, best_epoch, waited = math.inf, None, 0, 0
    started = time.time()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = generator.permutation(fit_races)
        total, batches = 0.0, 0
        for start in range(0, len(order), config.batch_races):
            batch = _batch(inputs, order[start:start + config.batch_races], device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                scores = model(batch)
            loss = pl_loss(scores, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss)
            batches += 1
        stop_loss = evaluate_loss(model, inputs, stop_races, device, config.batch_races * 4)
        report(event='nn_epoch', epoch=epoch, train_loss=round(total / batches, 5), stop_loss=round(stop_loss, 5),
               seconds=round(time.time() - started, 1))
        if stop_loss < best - 1e-5:
            best, best_epoch, waited = stop_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
            if waited >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(best_epoch=best_epoch, stop_loss=best, epochs=epoch, seconds=round(time.time() - started, 1),
                       device=device, config=asdict(config), ancestors=len(inputs.ancestor_vocab),
                       jockeys=len(inputs.jockey_vocab), trainers=len(inputs.trainer_vocab))


def evaluate_loss(model, inputs, races, device, batch_races):
    import torch
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(races), batch_races):
            chunk = races[start:start + batch_races]
            batch = _batch(inputs, chunk, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                scores = model(batch)
            total += float(pl_loss(scores, batch)) * len(chunk)
            count += len(chunk)
    return total / max(count, 1)


def predict(model, inputs, rows, *, device='auto', batch_races=1024):
    """Scores for ``rows`` (whole races are scored together; rows outside ``rows`` stay context-free)."""
    import numpy as np
    import torch
    device = next(model.parameters()).device.type if device == 'auto' else device
    races = inputs.races_for(rows, need_finishers=False)
    scores = np.full(len(inputs.race_of_row), np.nan)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(races), batch_races):
            batch = _batch(inputs, races[start:start + batch_races], device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                output = model(batch).float().cpu().numpy()
            index = batch['index']
            scores[index[index >= 0]] = output[index >= 0]
    return scores[rows]
