from datetime import datetime
import itertools
import logging
from pprint import pformat
import typing as t

from bgpsyche.logging_config import logging_setup
from bgpsyche.service.ext import ripe_ris, routeviews
from bgpsyche.service.ext.prefix2as import (
    AS2Prefix, Prefix2AS, get_prefix2as_caida, get_prefix2as_from_custom
)
from bgpsyche.util.net.typ import IPNetwork

# TODO: more better very much validation

logging_setup()
_LOG = logging.getLogger(__name__)


def _prefix2as(mapping: Prefix2AS, prefix: IPNetwork) -> t.Set[int]:
    if not prefix in mapping: return set()
    return mapping[prefix]


def _as2prefix(mapping: AS2Prefix, asn: int) -> t.Set[IPNetwork]:
    if not asn in mapping: return set()
    return mapping[asn]


def _from_ext(
        origin: t.Literal['routeviews', 'ris'], dt: datetime,
        collectors: t.List[str]
) -> t.Iterator[t.Tuple[int, IPNetwork]]:
    iter = routeviews.iter_paths_with_prefix(dt, collectors) \
        if origin == 'routeviews' \
        else ripe_ris.iter_paths_with_prefix(dt, collectors)

    return ((m['path'][-1], m['dst_prefix']) for m in iter)

def _research_prefix2as():

    p2a_caida, a2p_caida = get_prefix2as_caida(
        datetime.fromisoformat('2023-05-01T12:00')
    )

    p2a_custom, a2p_custom = get_prefix2as_from_custom(itertools.chain(
        # _from_ext(
        #     'ris', datetime.fromisoformat('2023-05-01T00:00'),
        #     ripe_ris.ACTIVE_COLLECTORS
        # ),
        _from_ext(
            'routeviews', datetime.fromisoformat('2023-05-01T00:00'),
            routeviews.COLLECTORS_BASE
        ),
    ))

    confirmed, changed = 0, 0

    missing_custom = set(p2a_caida).difference(set(p2a_custom))
    missing_caida = set(p2a_custom).difference(set(p2a_caida))

    for prefix in set(p2a_caida).union(p2a_custom):
        if p2a_caida[prefix] == p2a_custom[prefix]: confirmed += 1
        else: changed += 1

    _LOG.warning(f'Result: ' + pformat({
        'missing_custom': len(missing_custom), 'missing_caida': len(missing_caida),
        'missing_caida_v6': len([ p for p in p2a_custom if p.version == 6 ]),
        'confirmed': confirmed, 'changed': changed,
        'as_amnt_caida': len(a2p_caida), 'as_amnt_custom': len(a2p_custom),
    }))



if __name__ == '__main__': _research_prefix2as()