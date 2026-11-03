"""V3 experiment L: TabM-style neural strength model (design: TabM + periodic numerical embeddings + race attention).

Runs on the LightGBM cache of experiment K (``--cache-dir``; built on the first run if missing):

``nn_j``         experiment J's network (reference, its blend ``D_nn_j`` was J's best),
``nn_sched``     the same network with a completed one-cycle schedule,
``tabm``         32-member TabM ensemble + periodic numerical embeddings + attention across the runners,
``tabm_noattn``  tabm without the attention (each runner scored on its own features only),
``tabm_noemb``   tabm without the numerical embeddings.

Each is scored alone and blended with B (LightGBM + pedigree features); baselines A_lgbm and D_nn_j.
"""
from __future__ import annotations

import sys

from . import experiment_v3k

DEFAULT_VARIANTS = ('nn_j', 'nn_sched', 'tabm', 'tabm_noattn', 'tabm_noemb')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--variants' not in argv:
        argv += ['--variants', *DEFAULT_VARIANTS]
    if '--search-trials' not in argv:
        argv += ['--search-trials', '0']
    return experiment_v3k.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
