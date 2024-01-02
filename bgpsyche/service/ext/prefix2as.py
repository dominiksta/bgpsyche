from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import functools
import gzip
import ipaddress
import itertools
import typing as t

from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.service.ext import ripe_ris, routeviews
from bgpsyche.util.benchmark import bench_function
from bgpsyche.util.const import DATA_DIR
from bgpsyche.util.net.download import download_file_cached
from bgpsyche.util.net.typ import IPNetwork

Prefix2AS = t.Dict[IPNetwork, t.Set[int]]
AS2Prefix = t.Dict[int, t.Set[IPNetwork]]


# @bench_function
def prefix2as(prefix: IPNetwork) -> t.Set[int]:
    prefix2as_, _ = get_prefix2as_default()
    # tree_search = lookup['tree_v4'] \
    #     if prefix.version == 4 \
    #     else lookup['tree_v6']

    # found = tree_search.search(t.cast(t.Any, prefix.network_address))
    # if not found: return set()
    if not prefix in prefix2as_: return set()
    return prefix2as_[prefix]


# @bench_function
def as2prefix(asn: int) -> t.Set[IPNetwork]:
    _, as2prefix_ = get_prefix2as_default()
    if not asn in as2prefix_: return set()
    return as2prefix_[asn]


@functools.lru_cache()
@bench_function
def get_prefix2as_default(
        # HACK: compute datetime for newest
        caida_dt              = datetime.fromisoformat('2023-05-01T12:00'),
        ris_dt                = datetime.fromisoformat('2023-05-01T00:00'),
        ris_collectors        = ripe_ris.ACTIVE_COLLECTORS,
        routeviews_dt         = datetime.fromisoformat('2023-05-01T00:00'),
        routeviews_collectors = routeviews.COLLECTORS_BASE,
) -> t.Tuple[Prefix2AS, AS2Prefix]:
    @bench_function
    def _get_prefix2as_raw():
        from_caida = get_prefix2as_caida(caida_dt)
        from_custom = get_prefix2as_from_custom(
            itertools.chain(
                (
                    (m['path'][-1], m['dst_prefix'])
                    for m in ripe_ris.iter_paths_with_prefix(
                            ris_dt, ris_collectors
                    )
                ),
                (
                    (m['path'][-1], m['dst_prefix'])
                    for m in routeviews.iter_paths_with_prefix(
                            routeviews_dt, routeviews_collectors
                    )
                ),
            )
        )
        merged = functools.reduce(_merge_prefix2as, [
            from_caida, from_custom
        ])
        return merged

    cache = PickleFileCache('prefix2as', _get_prefix2as_raw)
    # cache.invalidate()
    return cache.get()


@bench_function
def _merge_prefix2as(
        a: t.Tuple[Prefix2AS, AS2Prefix],
        b: t.Tuple[Prefix2AS, AS2Prefix],
) -> t.Tuple[Prefix2AS, AS2Prefix]:
    prefix2as, as2prefix = deepcopy(a[0]), deepcopy(a[1])

    for prefix, asns in b[0].items():
        prefix2as[prefix] = prefix2as[prefix].union(asns)
    for asn, prefixes in b[1].items():
        as2prefix[asn] = as2prefix[asn].union(prefixes)

    return prefix2as, as2prefix


@bench_function
def get_prefix2as_from_custom(
        iter: t.Iterable[t.Tuple[int, IPNetwork]]
) -> t.Tuple[Prefix2AS, AS2Prefix]:
    prefix2as: Prefix2AS = defaultdict(set)
    as2prefix: AS2Prefix = defaultdict(set)

    for asn, prefix in iter:
        as2prefix[asn].add(prefix)
        prefix2as[prefix].add(asn)

    return prefix2as, as2prefix


@bench_function
def get_prefix2as_caida(
        dt: datetime,
        collector: str = 'rv2'
) -> t.Tuple[Prefix2AS, AS2Prefix]:
    """Get CAIDA's prefix2as mapping.

    According to their README, this mapping is computed from RouteViews MRT
    dumps through some perl script that is long considered deprecated.

    They also claim that they produce ipv6 mappings from the RouteViews
    collector rv6, but these do not seem to actually exist.

    Due to the lack of ipv6 alone, we need to agument this dataset ourselves.
    """
    prefix2as: Prefix2AS = defaultdict(set)
    as2prefix: AS2Prefix = defaultdict(set)

    base = 'https://publicdata.caida.org/datasets/routing/routeviews-prefix2as'
    path = f'{dt.year}/{dt.month:02d}'
    fname = f'routeviews-{collector}-{dt.year}{dt.month:02d}{dt.day:02d}' + \
        f'-{dt.hour:02d}{dt.minute:02d}.pfx2as.gz'

    path = download_file_cached(
        f'{base}/{path}/{fname}',
        DATA_DIR / 'caida_prefix2as' / fname
    )

    with gzip.open(path, 'rt', encoding='UTF-8') as f:
        for line in f:
            prefix_str, length, asns_str = [ l.strip() for l in line.split('\t')]
            # multi-origin as are delimited by '_'
            # asns in an as-set are delimited by ','
            # see https://publicdata.caida.org/datasets/routing/routeviews-prefix2as/
            asns = set(map(int, asns_str.replace('_', ',').split(',')))
            # they say that they do v6 as well, but i does not seem like they
            # actually still do that. this should alert us by failing if they
            # ever do v6 again.
            prefix = ipaddress.IPv4Network(f'{prefix_str}/{length}')
            assert prefix not in prefix2as

            for asn in asns: as2prefix[asn].add(prefix)
            prefix2as[prefix] = asns

    return prefix2as, as2prefix