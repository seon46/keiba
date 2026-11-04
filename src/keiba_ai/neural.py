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
    # TabM-style parameter-efficient ensemble (1 = a single network, the experiment J model)
    ensemble: int = 1
    # periodic numerical embeddings (each scaled feature -> learned-frequency sin/cos -> small vector)
    numeric_embedding: bool = False
    n_frequencies: int = 48
    frequency_scale: float = .01
    numeric_dim: int = 16
    # attention across the runners of a race
    attention: bool = True
    beta2: float = .999


class TrialTooSlow(RuntimeError):
    """Raised when a training run would take longer than its time limit."""


_CAPPED = []


def _cap_gpu_memory(torch):
    """Keep PyTorch within the GPU memory that is free now.  On Windows (WDDM) the driver otherwise
    spills into system RAM and a run becomes tens of times slower instead of failing with
    "out of memory"; with the cap an oversized setting fails fast and can be skipped."""
    if _CAPPED:
        return
    try:
        free, total = torch.cuda.mem_get_info()
        fraction = max(.3, min(.92, (free - 600 * 2 ** 20) / total))
        torch.cuda.set_per_process_memory_fraction(fraction)
        _CAPPED.append(fraction)
    except Exception:  # an older driver/runtime without mem_get_info: no cap
        _CAPPED.append(None)


def device_name(prefer='auto'):
    """'cuda' only after a real kernel ran on the GPU (a wheel without this GPU's architecture fails here)."""
    import warnings
    import torch
    if prefer == 'cpu':
        return 'cpu'
    if torch.cuda.is_available():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if float((torch.ones(4, device='cuda') * 2).sum()) == 8.0:
                    _cap_gpu_memory(torch)
                    return 'cuda'
        except Exception as exc:  # e.g. "no kernel image is available for execution on the device"
            if prefer == 'cuda' or prefer == 'auto':
                raise ValueError('GPUはありますが、このPyTorchはそのGPUに対応していません'
                                 f'（{torch.cuda.get_device_name(0)}）。scripts\\setup-nn.ps1 を実行し直してください。'
                                 f' 詳細: {exc}') from exc
    if prefer == 'cuda':
        raise ValueError('GPU（CUDA）を使えません。PyTorchのGPU版を入れてください（scripts\\setup-nn.ps1）。')
    return 'cpu'


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
        self.continuous = scaled.shape[1]  # the first columns are scaled values, the rest missing flags
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
        """Races containing any of ``rows`` (with >= 2 finishers when requested).  The whole race is
        always used, so non-finishers stay in the context even when ``rows`` lists finishers only."""
        import numpy as np
        races = np.unique(self.race_of_row[np.asarray(rows, dtype=np.int64)])
        if need_finishers:
            finishers = np.add.reduceat(self.finished.astype(np.int64), self.starts)
            races = races[finishers[races] >= 2]
        return races


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
    if config.ensemble > 1 or config.numeric_embedding or not config.attention:
        return build_tabm_model(inputs, config)
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


def build_tabm_model(inputs, config: NeuralConfig):
    """TabM-style network: shared weights + per-member multiplicative adapters (k = ``config.ensemble``),
    optional periodic numerical embeddings and optional attention across the runners.  Returns scores of
    shape (races, k, runners); the members are trained on their own PL losses and averaged to predict."""
    import math
    import torch
    from torch import nn

    k = max(1, config.ensemble)

    class EnsembleLinear(nn.Module):
        def __init__(self, width_in, width_out, random_first):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(width_in, width_out))
            nn.init.kaiming_uniform_(self.weight.T, a=math.sqrt(5))
            if random_first:
                self.r = nn.Parameter(torch.randint(0, 2, (k, width_in)).float() * 2 - 1)
            else:
                self.r = nn.Parameter(torch.ones(k, width_in))
            self.s = nn.Parameter(torch.ones(k, width_out))
            bound = 1 / math.sqrt(width_in)
            self.bias = nn.Parameter(torch.empty(width_out).uniform_(-bound, bound).repeat(k, 1))

        def forward(self, x):  # (..., k, in)
            return (x * self.r) @ self.weight * self.s + self.bias

    class Periodic(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.frequencies = nn.Parameter(torch.randn(features, config.n_frequencies) * config.frequency_scale)
            self.linear = nn.Linear(2 * config.n_frequencies, config.numeric_dim)

        def forward(self, x):  # (..., F) -> (..., F * numeric_dim)
            angle = 2 * math.pi * self.frequencies * x[..., None]
            embedded = torch.relu(self.linear(torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)))
            return embedded.flatten(-2)

    class TabRaceNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cat_embeddings = nn.ModuleList(nn.Embedding(card + 1, min(8, card + 1)) for card in inputs.cat_cards)
            self.jockey = nn.Embedding(len(inputs.jockey_vocab) + 1, config.embedding, padding_idx=0)
            self.trainer = nn.Embedding(len(inputs.trainer_vocab) + 1, config.embedding, padding_idx=0)
            self.ancestor = nn.Embedding(len(inputs.ancestor_vocab) + 1, config.pedigree_dim, padding_idx=0)
            generations = torch.tensor([len(path) for path in _paths()], dtype=torch.float32)
            self.path_weight = nn.Parameter(.5 ** generations)
            continuous = inputs.continuous
            flags = inputs.numeric.shape[1] - continuous
            self.periodic = Periodic(continuous) if config.numeric_embedding else None
            numeric_width = continuous * (config.numeric_dim if config.numeric_embedding else 1) + flags
            width = (numeric_width + sum(min(8, card + 1) for card in inputs.cat_cards)
                     + 2 * config.embedding + config.pedigree_dim)
            self.first = EnsembleLinear(width, config.width, True)
            self.second = EnsembleLinear(config.width, config.width, False)
            self.dropout = nn.Dropout(config.dropout)
            if config.attention:
                layer = nn.TransformerEncoderLayer(config.width, config.heads, 2 * config.width, config.dropout,
                                                   batch_first=True, norm_first=True, activation='gelu')
                self.context = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
            else:
                self.context = None
            self.norm = nn.LayerNorm(config.width)
            self.head_weight = nn.Parameter(torch.randn(k, config.width) / math.sqrt(config.width))
            self.head_bias = nn.Parameter(torch.zeros(k))

        def forward(self, batch):
            ancestry = batch['ancestry']
            if self.training and config.ancestor_dropout:
                ancestry = ancestry * (torch.rand(ancestry.shape, device=ancestry.device) >= config.ancestor_dropout)
            pedigree = (self.ancestor(ancestry) * self.path_weight[None, None, :, None]).sum(2)
            numeric = batch['numeric']
            if self.periodic is not None:
                numeric = torch.cat([self.periodic(numeric[..., :inputs.continuous]),
                                     numeric[..., inputs.continuous:]], dim=-1)
            parts = [numeric, pedigree, self.jockey(batch['jockey']), self.trainer(batch['trainer'])]
            parts += [embedding(batch['cat'][..., position]) for position, embedding in enumerate(self.cat_embeddings)]
            x = torch.cat(parts, dim=-1)                                  # (B, N, D)
            races, runners = x.shape[0], x.shape[1]
            x = x.unsqueeze(2).expand(races, runners, k, x.shape[-1])     # (B, N, k, D)
            hidden = self.dropout(nn.functional.gelu(self.first(x)))
            hidden = nn.functional.gelu(self.second(hidden))              # (B, N, k, W)
            hidden = hidden.permute(0, 2, 1, 3)                           # (B, k, N, W)
            if self.context is not None:
                flat = hidden.reshape(races * k, runners, hidden.shape[-1])
                mask = (~batch['mask']).repeat_interleave(k, dim=0)
                hidden = self.context(flat, src_key_padding_mask=mask).reshape(races, k, runners, -1)
            hidden = self.norm(hidden)
            return torch.einsum('bknw,kw->bkn', hidden, self.head_weight) + self.head_bias[None, :, None]

    return TabRaceNet()


def _paths():
    from .pedigree import PATHS
    return PATHS


def member_loss(scores, batch):
    """PL loss of a (races, runners) score matrix, or the mean over members of (races, k, runners)."""
    if scores.dim() == 2:
        return pl_loss(scores, batch)
    races, k, runners = scores.shape
    expand = lambda tensor: tensor[:, None, :].expand(races, k, runners).reshape(races * k, runners)
    return pl_loss(scores.reshape(races * k, runners), dict(finished=expand(batch['finished']),
                                                           positions=expand(batch['positions'])))


def ensemble_scores(scores):
    return scores if scores.dim() == 2 else scores.mean(1)


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

def train(inputs, fit_rows, stop_rows, config: NeuralConfig, *, device='auto', report=None, time_limit=None):
    """``time_limit`` (seconds): stop with TrialTooSlow as soon as the projected run time exceeds it."""
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                  betas=(.9, config.beta2))
    steps = math.ceil(len(fit_races) / config.batch_races) * config.max_epochs
    scheduler = (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=steps,
                                                     pct_start=.1) if steps >= 50 else None)
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
            loss = member_loss(scores, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item()
            batches += 1
            if time_limit and epoch == 1 and batches == 20:
                per_batch = (time.time() - started) / batches
                projected = per_batch * math.ceil(len(fit_races) / config.batch_races) * config.max_epochs
                if projected > time_limit:
                    raise TrialTooSlow(f'推定{projected / 60:.0f}分かかるため打ち切りました。')
            if time_limit and time.time() - started > time_limit:
                raise TrialTooSlow('時間の上限を超えたため打ち切りました。')
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
                scores = ensemble_scores(model(batch))
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
                output = ensemble_scores(model(batch)).float().cpu().numpy()
            index = batch['index']
            scores[index[index >= 0]] = output[index >= 0]
    return scores[rows]
