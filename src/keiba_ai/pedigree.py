"""Five-generation pedigrees from the scraper database, and causal pedigree features (design E).

Ancestor keys are the netkeiba horse IDs (or ``name:<name>`` when only a name
is known).  ``PATHS`` orders the 62 ancestors: S, D, SS, SD, DS, DD, ...
(S = sire side, D = dam side, read from the horse outward).

Pedigree features reuse the day-by-day Gaussian Plackett-Luce update of the
jockey effect (``experiment_v3d.jockey_timeline``) with the sire, the dam's
sire and the dam as the entity and an out-of-fold horse-strength offset: a
sire's theta measures how much its offspring ran *above* the strength the
model already gave them.  Every row sees only earlier days.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

PATHS = tuple(''.join('D' if bit == '1' else 'S' for bit in f'{branch:0{generation}b}')
              for generation in range(1, 6) for branch in range(2 ** generation))
PEDIGREE_FEATURES = (
    'sire_theta', 'sire_theta_sd', 'sire_runs', 'sire_theta_race_z',
    'sire_surface_theta', 'sire_surface_runs', 'sire_distance_theta', 'sire_distance_runs',
    'damsire_theta', 'damsire_runs', 'damsire_surface_theta', 'damsire_surface_runs',
    'dam_theta', 'dam_runs', 'pedigree_known',
)


def load_pedigrees(race_db, horse_ids=None):
    """{horse_id: tuple of 62 ancestor keys (None = unknown)} from the latest pedigree snapshot of each horse."""
    path = Path(race_db)
    if not path.is_file():
        raise ValueError(f'収集DBがありません: {path}')
    wanted = set(horse_ids) if horse_ids is not None else None
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
    try:
        rows = connection.execute('SELECT p.horse_id,p.metadata_json FROM pedigrees p JOIN observations o '
                                  'ON o.id=p.snapshot_id ORDER BY o.retrieved_at,o.rowid')
        latest = {}
        for horse, metadata in rows:
            if wanted is None or horse in wanted:
                latest[horse] = metadata
    except sqlite3.Error as exc:
        raise ValueError(f'収集DBに血統の表がありません: {exc}') from exc
    finally:
        connection.close()
    pedigrees = {}
    for horse, metadata in latest.items():
        by_path = {item.get('path'): item for item in json.loads(metadata).get('ancestors', [])}
        keys = []
        for ancestor_path in PATHS:
            item = by_path.get(ancestor_path) or {}
            identifier = item.get('horse_id') or (f'name:{item["name"]}' if item.get('name') else None)
            keys.append(identifier)
        pedigrees[horse] = tuple(keys)
    return pedigrees


def distance_band(distance):
    import math
    if distance is None or not math.isfinite(distance):
        return ''
    return 's' if distance < 1400 else 'm' if distance <= 1800 else 'i' if distance <= 2200 else 'l'


def pedigree_block(days, race_ids, horses, finished, positions, strength, surfaces, distances, pedigrees, *,
                   half_life=1095.0, short_half_life=180.0, prior_sd=.3):
    """Rows x PEDIGREE_FEATURES (float32) from causal sire / damsire / dam effects."""
    import numpy as np
    from .experiment_v3c import _race_z
    from .experiment_v3d import jockey_timeline
    count = len(days)
    index = {path: position for position, path in enumerate(PATHS)}
    sire = np.full(count, '', dtype=object)
    damsire = np.full(count, '', dtype=object)
    dam = np.full(count, '', dtype=object)
    known = np.zeros(count, dtype=np.float32)
    for row, horse in enumerate(horses.tolist()):
        keys = pedigrees.get(horse)
        if keys is None:
            continue
        sire[row] = keys[index['S']] or ''
        dam[row] = keys[index['D']] or ''
        damsire[row] = keys[index['DS']] or ''
        known[row] = sum(key is not None for key in keys) / len(PATHS)
    surface = np.asarray([str(int(value)) if np.isfinite(value) else '' for value in surfaces.astype(float)], dtype=object)
    band = np.asarray([distance_band(value) for value in distances.astype(float)], dtype=object)

    def joined(entity, suffix):
        return np.asarray([f'{a}|{b}' if a and b else '' for a, b in zip(entity.tolist(), suffix.tolist())], dtype=object)

    def run(entity):
        return jockey_timeline(days, race_ids, entity, horses, finished, positions, strength, half_life=half_life,
                               short_half_life=short_half_life, prior_sd=prior_sd)

    sire_all, sire_surface, sire_distance = run(sire), run(joined(sire, surface)), run(joined(sire, band))
    damsire_all, damsire_surface, dam_all = run(damsire), run(joined(damsire, surface)), run(dam)
    columns = [
        sire_all['theta'], sire_all['theta_sd'], sire_all['rides'], _race_z(sire_all['theta'], race_ids),
        sire_surface['theta'], sire_surface['rides'], sire_distance['theta'], sire_distance['rides'],
        damsire_all['theta'], damsire_all['rides'], damsire_surface['theta'], damsire_surface['rides'],
        dam_all['theta'], dam_all['rides'], known,
    ]
    return np.column_stack(columns).astype(np.float32)
