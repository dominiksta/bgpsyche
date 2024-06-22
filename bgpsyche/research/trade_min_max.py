from collections import defaultdict
from itertools import combinations
import logging
from pprint import pprint
import typing as t

from bgpsyche.logging_config import logging_setup
from bgpsyche.stage2_enrich.global_trade import get_trade_relationships
from bgpsyche.util.geo import ALPHA2_OFFICIAL

logging_setup()
_LOG = logging.getLogger(__name__)

def _research_trade_min_max():
    trade = get_trade_relationships()

    vals: t.Dict[str, float] = defaultdict(int)

    for c1, c2 in combinations(ALPHA2_OFFICIAL, r=2):
        for direction in ['imports', 'exports']:
            if c1 not in trade[direction] or c2 not in trade[direction][c1]: continue
            vals[f'{c1}<->{c2}'] += trade[direction][c1][c2]
            assert trade[direction][c1][c2] >= 0, \
                (f'{c1}<->{c2}', direction, trade[direction][c1][c2])
            

    _LOG.info(f'max: {max(vals.values())}, min: {min(vals.values())}')
    pprint(vals)


if __name__ == '__main__': _research_trade_min_max()