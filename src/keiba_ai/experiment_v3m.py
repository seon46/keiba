"""V3 experiment M: tuning the TabM network and blending several networks with LightGBM.

Runs on experiment K's LightGBM cache (``--cache-dir``):

``tabm``, ``tabm_noattn``   experiment L's settings (references),
``tabm_search``             per fold, a random search of TabM settings (``--search-trials``, attention on/off,
                            width, depth, regularisation, numerical-embedding frequencies, ensemble size ...)
                            on the stop-period PL loss; the best setting trained once,
``tabm_search_x3``          the best setting averaged over 3 seeds,
``D_multi``                 B blended with all of the networks above at once (weights on the stop period).

Each network is also scored alone and as ``D_<name>`` (blend with B).  Baselines A_lgbm and D_nn_j.
"""
from __future__ import annotations

import sys

from . import experiment_v3k

DEFAULT_VARIANTS = ('nn_j', 'tabm', 'tabm_noattn', 'tabm_search', 'tabm_search_x3')


def main(argv=None):
    from .priority import background
    background()
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--variants' not in argv:
        argv += ['--variants', *DEFAULT_VARIANTS]
    if '--search-trials' not in argv:
        argv += ['--search-trials', '12']
    return experiment_v3k.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
