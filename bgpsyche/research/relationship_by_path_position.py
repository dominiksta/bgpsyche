from collections import defaultdict
from datetime import datetime
import itertools
from pprint import pformat
import typing as t
import logging

from matplotlib import pyplot as plt
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.service.ext import routeviews, ripe_ris
from bgpsyche.service.ext.caida_asrel import get_caida_asrel
from bgpsyche.util.bgp.relationship import RelationshipKind
from bgpsyche.util.run_in_pypy import run_in_pypy

logging_setup()
_LOG = logging.getLogger(__name__)

@run_in_pypy(cache=PickleFileCache)
def _research_relationship_by_path_position():
    dt = datetime.fromisoformat('2023-05-01T00:00')

    rels = get_caida_asrel(dt)

    counts: t.Dict[int, t.Dict[t.Optional[RelationshipKind], int]] = \
        defaultdict(lambda: { 'p2p': 0, 'c2p': 0, 'p2c': 0, None: 0 })

    for path_meta in itertools.chain(
            routeviews.iter_paths(dt, eliminate_path_prepending=True),
            ripe_ris.iter_paths(dt, eliminate_path_prepending=True),
    ):
        path = path_meta['path']
        for i in range(0, len(path) - 1):
            src, sink = path[i], path[i+1]
            if src not in rels or sink not in rels[src]:
                counts[i][None] += 1
                continue
            counts[i][rels[src][sink]] += 1


    probs: t.Dict[int, t.Dict[t.Optional[RelationshipKind], float]] = defaultdict(dict)

    for pos, rel2count in counts.items():
        count_sum = sum(rel2count.values())
        for rel, count in rel2count.items():
            probs[pos][rel] = round(count / count_sum, 2)

    return probs
            


if __name__ == '__main__':
    probs = _research_relationship_by_path_position()
    plt.plot(
        list(probs.keys()), [ pos_prob['p2c'] for pos_prob in probs.values() ],
        label='p2c',
    )
    plt.plot(
        list(probs.keys()), [ pos_prob['c2p'] for pos_prob in probs.values() ],
        label='c2p'
    )
    plt.plot(
        list(probs.keys()), [ pos_prob['p2p'] for pos_prob in probs.values() ],
        label='p2p',
    )
    plt.plot(
        list(probs.keys()), [ pos_prob[None] for pos_prob in probs.values() ],
        label='Unknown',
    )
    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('Link Position in Path')
    plt.show()
    _LOG.info(pformat(probs))
    