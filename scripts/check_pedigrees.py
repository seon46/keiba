"""Read-only check of collected pedigrees against the prepared features database.

Usage (PowerShell, from the repository root):
  .\\.venv\\Scripts\\python.exe -X utf8 scripts\\check_pedigrees.py --dataset data\\ml_rank_bias_v2\\features.sqlite3 --output data\\pedigree_check.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)

    store = connect(args.data_dir / 'keiba.sqlite3')
    latest = {}
    snapshots = 0
    for horse, retrieved, metadata in store.execute(
            'SELECT p.horse_id,o.retrieved_at,p.metadata_json FROM pedigrees p JOIN observations o ON o.id=p.snapshot_id '
            'ORDER BY o.retrieved_at'):
        snapshots += 1
        latest[horse] = (retrieved, metadata)
    jobs = []
    try:
        for row in store.execute("SELECT * FROM collection_jobs"):
            jobs.append(row)
    except sqlite3.Error:
        pass
    store.close()

    known_counts, identified_counts, generation_known = Counter(), Counter(), defaultdict(Counter)
    sires, dam_sires = Counter(), Counter()
    key_paths = {'S': 0, 'D': 0, 'SS': 0, 'DS': 0, 'S_id': 0, 'D_id': 0, 'DS_id': 0}
    dates = []
    for horse, (retrieved, metadata) in latest.items():
        data = json.loads(metadata)
        dates.append(retrieved[:10])
        known_counts[min(62, data.get('known_ancestor_count') or 0) // 10 * 10] += 1
        identified_counts[min(62, data.get('identified_ancestor_count') or 0) // 10 * 10] += 1
        by_path = {item['path']: item for item in data.get('ancestors', [])}
        for item in by_path.values():
            generation_known[item['generation']]['known' if (item.get('name') or item.get('horse_id')) else 'unknown'] += 1
        for path in ('S', 'D', 'SS', 'DS'):
            if by_path.get(path, {}).get('name'):
                key_paths[path] += 1
        for path in ('S', 'D', 'DS'):
            if by_path.get(path, {}).get('horse_id'):
                key_paths[path + '_id'] += 1
        if by_path.get('S', {}).get('name'):
            sires[by_path['S']['name']] += 1
        if by_path.get('DS', {}).get('name'):
            dam_sires[by_path['DS']['name']] += 1

    features = connect(args.dataset)
    coverage = defaultdict(lambda: [0, 0])
    horses_by_group = defaultdict(lambda: [set(), set()])
    for group, day, horse in features.execute('SELECT group_name,day,horse_id FROM features'):
        key = f'{group}/{day[:4]}'
        coverage[key][0] += 1
        coverage[key][1] += horse in latest
        horses_by_group[group][0].add(horse)
        if horse in latest:
            horses_by_group[group][1].add(horse)
    features.close()

    total = len(latest)
    report = dict(
        pedigree_snapshots=snapshots, horses_with_pedigree=total,
        retrieved_first=min(dates) if dates else None, retrieved_last=max(dates) if dates else None,
        key_ancestors_known={key: round(value / total, 4) if total else None for key, value in key_paths.items()},
        known_ancestor_count_hist={f'{k}-{k + 9}': v for k, v in sorted(known_counts.items())},
        identified_ancestor_count_hist={f'{k}-{k + 9}': v for k, v in sorted(identified_counts.items())},
        known_share_by_generation={g: round(c['known'] / max(1, c['known'] + c['unknown']), 4)
                                   for g, c in sorted(generation_known.items())},
        runner_rows_with_pedigree={key: dict(rows=rows, with_pedigree=hit, share=round(hit / rows, 4))
                                   for key, (rows, hit) in sorted(coverage.items())},
        horses_with_pedigree_by_group={group: dict(horses=len(all_), with_pedigree=len(hit),
                                                   share=round(len(hit) / max(1, len(all_)), 4))
                                       for group, (all_, hit) in sorted(horses_by_group.items())},
        distinct_sires=len(sires), distinct_dam_sires=len(dam_sires),
        top_sires=sires.most_common(15), top_dam_sires=dam_sires.most_common(15),
        collection_jobs=len(jobs),
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(event='pedigree_check', output=str(args.output), horses_with_pedigree=total,
                          by_group=report['horses_with_pedigree_by_group']), ensure_ascii=False))


if __name__ == '__main__':
    main()
