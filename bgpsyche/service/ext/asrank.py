import bz2
from datetime import date
import logging
from pprint import pprint
import typing as t
from functools import lru_cache
import logging

from gql import gql as _gql
import gql
import gql.transport.requests
from bgpsyche.caching.json import JSONFileCache
from bgpsyche.util.benchmark import bench_function
from bgpsyche.util.const import DATA_DIR
from bgpsyche.util.net.download import download_file_cached

_LOG = logging.getLogger(__name__)

# ranking
# ----------------------------------------------------------------------

gql.transport.requests.log.setLevel(logging.WARNING)

def _req_asrank(query: str) -> t.Any:
    @lru_cache()
    def get_client():
        transport = gql.transport.requests.RequestsHTTPTransport(
            url='https://api.asrank.caida.org/v2/graphql'
        )
        return gql.Client(transport=transport, fetch_schema_from_transport=True)

    return get_client().execute(_gql(query))


class ASRank(t.TypedDict):
    sorted: t.List[int]
    by_asn: t.Dict[int, int]

@lru_cache()
def get_asrank_full() -> ASRank:
    out: ASRank = { 'by_asn': {}, 'sorted': [] }
    batch_size = 5_000
    has_next, current_offset = True, 0
    while has_next:
        _LOG.info(
            f'Getting ASRank offset {current_offset}, batch_size {batch_size}'
        )
        resp = JSONFileCache(
            f'asrank_first_{batch_size}_offset_{current_offset}',
            lambda: _req_asrank("""{
              asns(first: %d, offset: %d, sort: "+rank") {
                pageInfo {
                  first,
                  offset,
                  hasNextPage,
                },
                edges {
                  node {
                    asn
                  }
                }
              }
            }""" % (batch_size, current_offset))
        ).get()
        asns = [ int(el['node']['asn']) for el in resp['asns']['edges'] ]

        for i in range(0, len(asns)):
            out['sorted'].append(asns[i])
            out['by_asn'][asns[i]] = current_offset + i + 1

        has_next = resp['asns']['pageInfo']['hasNextPage']
        current_offset += batch_size

    assert len(out['sorted']) > 100_000
    return out


def get_asrank(asn: int) -> int:
    asrank = get_asrank_full()
    return asrank['by_asn'][asn] if asn in asrank['by_asn'] else -1


# customer cones
# ----------------------------------------------------------------------

_CUSTOMER_CONE_DOWNLOAD_DIR = DATA_DIR / 'asrank_customer_cones'
_CUSTOMER_CONE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache()
@bench_function
def get_asrank_customer_cones_full(
        dt: date = date.fromisoformat('2023-05-01'),
) -> t.Dict[int, t.Set[int]]:
    ret: t.Dict[int, t.Set[int]] = {}
    fname = f'{dt.strftime("%Y%m%d")}.ppdc-ases.txt.bz2'
    file = download_file_cached(
        f'http://data.caida.org/datasets/as-relationships/serial-1/{fname}',
        _CUSTOMER_CONE_DOWNLOAD_DIR / fname
    )

    for line in bz2.open(file, 'rt', encoding='utf-8'):
        if line.startswith('#'): continue
        ases = [ int(asn) for asn in line.split(' ') ]
        ret[ases[0]] = set(ases[1:]).difference({ases[0]})

    return ret


def get_asrank_customer_cones(
        asn: int,
        dt: date = date.fromisoformat('2023-05-01'),
) -> t.Set[int]:
    cones = get_asrank_customer_cones_full(dt)
    return cones[asn] if asn in cones else set()


@lru_cache()
def get_asrank_customer_cone_sizes_full(
        dt: date = date.fromisoformat('2023-05-01'),
) -> t.Dict[int, int]:
    cones = get_asrank_customer_cones_full(dt)
    return { asn: len(cone) for asn, cone in cones.items() }


def get_asrank_customer_cone_size(
        asn: int,
        dt: date = date.fromisoformat('2023-05-01'),
) -> int:
    cones = get_asrank_customer_cone_sizes_full(dt)
    return cones[asn] if asn in cones else 0


ASRANK_CUSTOMER_CONE_SIZE_RANGE: t.Tuple[int, int] = (
    min(get_asrank_customer_cone_sizes_full().values()),
    max(get_asrank_customer_cone_sizes_full().values()),
)

if __name__ == '__main__':
    show = {
        3356: 'Level3',
        3257: 'GTT',
        1239: 'Sprint',
        3320: 'DTAG',
        39063: 'Leitwert',
        51402: 'COM-IN',
        51378: 'Klinikum Ingolstadt',
        16509: 'Amazon',
        64199: 'TCPShield (DDOS Protection)',
        13335: 'Cloudflare',
        17374: 'Walmart',
        32934: 'Meta (Zuckbook)',
        8075: 'Micro$oft',
        6695: 'DE-CIX Frankfurt Route Servers',
    }

    pprint({
        asn: {
            'name': show[asn],
            'cone': get_asrank_customer_cone_size(asn),
            'rank': get_asrank(asn)
        }
        for asn in show.keys()
    })