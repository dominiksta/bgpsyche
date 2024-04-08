from datetime import datetime
from functools import lru_cache
import logging
import typing as t

from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.service.mrt_file_parser import ASPathMeta
from bgpsyche.service.ext import ripe_ris

logging_setup()
_LOG = logging.getLogger(__name__)

_Src2Dst2Count = t.Dict[int, t.Dict[int, int]]

# we could do prefix destinations in the future, but lets start with asns
BGPMarkovChainsPerDest = t.Dict[int, _Src2Dst2Count]


def bgp_markov_chains_from_mrt_dumps(
        as_paths_with_count: t.Iterator[ASPathMeta],
) -> BGPMarkovChainsPerDest:
    ret: BGPMarkovChainsPerDest = {}

    for meta in as_paths_with_count:
        dst_asn = meta['path'][-1]
        for i in range(len(meta['path']) - 1):
            src, dst = meta['path'][i], meta['path'][i+1]
            if dst_asn not in ret: ret[dst_asn] = {}
            if src not in ret[dst_asn]: ret[dst_asn][src] = {}
            if dst not in ret[dst_asn][src]: ret[dst_asn][src][dst] = 0
            ret[dst_asn][src][dst] += meta['count']

    return ret


def get_as_path_confidence(
        as_path: t.List[int],
        chains: BGPMarkovChainsPerDest,
) -> float:
    """Compute a confidence value from -1 to 1 for a given AS path.

    A confidence value of ...
    - ... 1 encodes high certainty that the path is correct.
    - ... 0 encodes unsure or no opinion.
    - ...-1 encodes high certainty that the path is not correct.
    """
    if as_path[-1] not in chains:
        _LOG.warning(f'No chain for destination: {as_path}')
        return 0

    chain = chains[as_path[-1]]
    confidence_path: t.List[float] = []
    confidence_path_str: str = ''

    for i in range(len(as_path) - 1):
        src, dst = as_path[i], as_path[i+1]
        confidence = 0

        if src not in chain or \
           dst not in chain[src]:
            _LOG.debug(f'Missing in markov chain: {(src, dst)}')
        else:
            counts_sum = sum(chain[src].values())
            prob = chain[src][dst] / counts_sum
            direction = (prob - 0.5) * 2 # [-1;1]
            confidence_min = 5
            counts_confidence = min(counts_sum, confidence_min) \
                / confidence_min # [0;1]
            confidence = round(direction * counts_confidence, 2) # [-1;1]
            # print(counts_sum, prob, direction, counts_confidence, confidence)

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
) -> BGPMarkovChainsPerDest:
    cache = PickleFileCache(
        f'bgp_markov_chain_ris_{dt.strftime("%Y%m%d.%H%M")}',
        lambda: bgp_markov_chains_from_mrt_dumps(ripe_ris.iter_paths(
            dt, eliminate_path_prepending=True
        ))
    )

    # cache.invalidate()
    return cache.get()


def _test_ris(path: t.List[int]):
    chain = markov_chain_from_ripe_ris(datetime.fromisoformat('2023-05-01T00:00'))

    # print(pformat(chain[path[-1]][19151]))
    print(get_as_path_confidence(path, chain))


if __name__ == '__main__':
    path = list(map(int, '29680 3257 3320'.split(' ')))

    _test_ris(path)
    # _test_atlas(path)