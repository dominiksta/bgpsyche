from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from itertools import pairwise
import logging
import typing as t

from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.service.mrt_file_parser import ASPathMeta
from bgpsyche.service.ext import ripe_ris

logging_setup()
_LOG = logging.getLogger(__name__)

_LinkSeenCount = t.Dict[int, t.Dict[int, int]]

# we could do prefix destinations in the future, but lets start with asns
_LinkSeenCountPerDest = t.Dict[int, _LinkSeenCount]


def bgp_markov_chains_from_mrt_dumps(
        as_paths_with_count: t.Iterator[ASPathMeta],
) -> t.Tuple[_LinkSeenCount, _LinkSeenCountPerDest]:
    per_dest: _LinkSeenCountPerDest = {}
    full: _LinkSeenCount = {}
    dbg_all_links: t.Set[t.Tuple[int, int]] = set()

    for meta in as_paths_with_count:
        dst_asn = meta['path'][-1]
        for as1, as2 in pairwise(meta['path']): dbg_all_links.add((as1, as2))

        for i in range(len(meta['path']) - 1):
            src, dst = meta['path'][i], meta['path'][i+1]

            if src not in full: full[src] = {}
            if dst not in full[src]: full[src][dst] = 0
            full[src][dst] += meta['count']

            if dst not in full: full[dst] = {}
            if src not in full[dst]: full[dst][src] = 0
            full[dst][src] += meta['count']

            if dst_asn not in per_dest: per_dest[dst_asn] = {}
            if src not in per_dest[dst_asn]: per_dest[dst_asn][src] = {}
            if dst not in per_dest[dst_asn][src]: per_dest[dst_asn][src][dst] = 0
            per_dest[dst_asn][src][dst] += meta['count']

    _LOG.info(
        f'Created BGP markov chain with {len(full)} ASes ' +
        f'and {len(dbg_all_links)} links'
    )

    return full, per_dest


def get_link_confidence(
        dst: int,
        link_counts_at_src: t.Dict[int, int],
) -> float: # [-1;1]
    """Compute a confidence value from -1 to 1 for a given AS pair.

    A confidence value of ...
    - ... 1 encodes high certainty that the link is correct.
    - ... 0 encodes unsure or no opinion.
    - ...-1 encodes high certainty that the link is not correct.
    """
    if len(link_counts_at_src.keys()) == 0: return 0

    CONFIDENCE_MIN = 5

    counts_sum = sum(link_counts_at_src.values())
    base_prob = 1 / len(link_counts_at_src.keys())

    # probability of link [0;1]
    prob = (
        (link_counts_at_src[dst] / counts_sum)
        if dst in link_counts_at_src else 0
    )

    delta = prob - base_prob

    # scale delta to [-1;1] using base_prob
    if base_prob == 1:
        direction = 1
    else:
        direction = (
            delta * (1 / base_prob)
            if delta < 0
            else delta * (1 / (1 - base_prob))
        )

    # [0;1]: if we have less then N links to compute prob from, this number goes
    # to 0
    counts_confidence = \
        min(counts_sum, CONFIDENCE_MIN) / CONFIDENCE_MIN

    return round(direction * counts_confidence, 2) # [-1;1]


def get_as_path_confidence(
        as_path: t.List[int],
        counts: _LinkSeenCount,
) -> float:
    """Compute a confidence value from -1 to 1 for a given AS path.

    A confidence value of ...
    - ... 1 encodes high certainty that the path is correct.
    - ... 0 encodes unsure or no opinion.
    - ...-1 encodes high certainty that the path is not correct.
    """
    confidence_path: t.List[float] = []
    confidence_path_str: str = ''

    for i in range(len(as_path) - 1):
        src, dst = as_path[i], as_path[i+1]
        confidence = 0

        if src not in counts or \
           dst not in counts[src]:
            _LOG.debug(f'Missing in markov chain: {(src, dst)}')
        else:
            confidence = get_link_confidence(dst, counts[src])

        confidence_path.append(confidence)
        confidence_path_str += f'{src} -{confidence}- '

    confidence_path_str += f'{as_path[-1]}'
    _LOG.debug(confidence_path_str)

    if len(confidence_path) == 0: return 0
    return sum(confidence_path)/len(confidence_path)


# from bgp
# ----------------------------------------------------------------------

@lru_cache()
def markov_chain_from_ripe_ris(
        dt: datetime
) -> t.Tuple[_LinkSeenCount, _LinkSeenCountPerDest]:
    cache = PickleFileCache(
        f'bgp_markov_chain_ris_{dt.strftime("%Y%m%d.%H%M")}',
        lambda: bgp_markov_chains_from_mrt_dumps(ripe_ris.iter_paths(
            dt, eliminate_path_prepending=True
        ))
    )

    # cache.invalidate()
    return cache.get()


def _test_ris(path: t.List[int]):
    full, per_dest = \
        markov_chain_from_ripe_ris(datetime.fromisoformat('2023-05-01T00:00'))

    # print(pformat(chain[path[-1]][19151]))
    print(get_as_path_confidence(path, full))


if __name__ == '__main__':
    path = list(map(int, '29680 3257 3320'.split(' ')))

    _test_ris(path)
    # _test_atlas(path)