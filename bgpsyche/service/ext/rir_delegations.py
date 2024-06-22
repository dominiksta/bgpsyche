from collections import defaultdict
from datetime import date, datetime, timedelta
import functools
import math
import logging
import typing as t
import urllib.request
import bz2
import gzip
import csv

from bgpsyche.caching import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.util.const import DATA_DIR
from bgpsyche.util.geo import ALPHA2_WITH_LOCATION, COUNTRY_UNKNOWN, Alpha2WithLocation

_BASE_URL = 'https://ftp.ripe.net/pub/stats'

RIR = t.Literal['afrinic', 'apnic', 'arin', 'lacnic', 'ripencc']
RIRS: t.Set[RIR] = set(t.get_args(RIR))

_DATA_DIR = DATA_DIR / 'rir_delegations'
_TMP_DIR = _DATA_DIR / 'tmp'

logging_setup()
_LOG = logging.getLogger()

def _iter_delegation_file(
        rir: RIR, dt: t.Union[t.Literal['latest'], date]
) -> t.Iterator[str]:
    _TMP_DIR.mkdir(parents=True, exist_ok=True)

    if dt == 'latest': dt = date.today() - timedelta(days=1)

    # URL/File format:
    #
    # - one per day
    # - afrinic:          {year}/delegated-{rir}-extended-{year}{month}{day}
    # - apnic:            {year}/delegated-{rir}-extended-{year}{month}{day}.gz
    # - arin:
    #   - current year:   delegated-{rir}-extended-{year}{month}{day}
    #   - previous years: archive/{year}/delegated-{rir}-extended-{year}{month}{day}.gz
    # - lacnic:           delegated-{rir}-extended-{year}{month}{day}
    # - ripencc:          {year}/delegated-{rir}-extended-{year}{month}{day}.bz2
    #
    # ....yeah

    df = lambda f: dt.strftime(f)
    fbase = f'delegated-{rir}-extended'

    url = f'{_BASE_URL}/{rir}/'
    fname = ''
    if rir == 'afrinic': url += f'{df("%Y")}/{fbase}-{df("%Y%m%d")}'
    elif rir == 'apnic': url += f'{df("%Y")}/{fbase}-{df("%Y%m%d")}.gz'
    elif rir == 'arin':
        if dt.year == date.today().year: url += f'{fbase}-{df("%Y%m%d")}'
        else: url += f'archive/{df("%Y")}/{fbase}-{df("%Y%m%d")}.gz'
    elif rir == 'lacnic': url += f'{fbase}-{df("%Y%m%d")}'
    elif rir == 'ripencc': url += f'{df("%Y")}/{fbase}-{df("%Y%m%d")}.bz2'

    fname = f'{fbase}-{df("%Y%m%d")}'
    if '.gz' in url or '.bz2' in url != 1: fname += '.' + url.split('.')[-1]
    tmp_path = _TMP_DIR / rir / fname

    tmp_path.parent.mkdir(exist_ok=True, parents=True)
    if not tmp_path.exists():
        _LOG.info(f'Downlading {url}')
        urllib.request.urlretrieve(url, tmp_path)
    else:
        _LOG.info(f'Using cached {url}')

    if url.endswith('.bz2'):
        with bz2.open(tmp_path, 'rt', encoding='utf-8') as f:
            for l in f: yield l
    elif url.endswith('.gz'):
        with gzip.open(tmp_path, 'rt', encoding='utf-8') as f:
            for l in f: yield l
    else:
        with open(tmp_path, 'r', encoding='utf-8') as f:
            for l in f: yield l


class _AssignmentLineRaw(t.TypedDict):
    registry: RIR
    cc: Alpha2WithLocation
    type: t.Literal['asn', 'ipv4', 'ipv6']
    start: str # first asn or ip address of block
    # for ipv4 and asn: number of addrs
    # for ipv6: cidr prefix length
    value: int
    date: t.Optional[date]
    status: t.Literal['allocated', 'assigned']
    # technically not standardized, but everyone has an id to associate an asn
    # assignment with ip assignments
    id: t.Optional[str]


def _parse(iter: t.Iterator[str]) -> t.Iterator[_AssignmentLineRaw]:

    # See https://ftp.ripe.net/pub/stats/ripencc/RIR-Statistics-Exchange-Format.txt
    # for format description

    def _csv_iter() -> t.Iterator[str]:
        for line in iter:
            if line.startswith('#'): continue
            if line.endswith('|summary\n'): continue
            if line.startswith('2'):
                header = line
                _LOG.info(f'Found header: {header[:-1]}')
                version = header.split('|')[0]
                if not version.startswith('2'):
                    ValueError(f'Invalid Format Version {version}')
                continue
                
            yield line

    for line in csv.reader(_csv_iter(), delimiter='|'):
        try:
            cc = t.cast(
                Alpha2WithLocation,
                COUNTRY_UNKNOWN if line[1] in ['', 'ZZ'] else line[1]
            )
            dt = None if (line[5] == '' or line[5] == '00000000') \
                else datetime.strptime(line[5], '%Y%m%d').date()
            if line[6] in {'available', 'reserved'}: continue
            ret: _AssignmentLineRaw = {
                'registry': t.cast(RIR, line[0]),
                'cc': cc,
                'type': t.cast(t.Literal['asn', 'ipv4', 'ipv6'], line[2]),
                'start': line[3],
                'value': int(line[4]),
                'date': dt,
                'status': t.cast(t.Literal['allocated', 'assigned'], line[6]),
                'id': (line[0] + '--' + line[7]) if len(line) >= 8 else None
            }
            assert ret['registry'] in RIRS, ret
            assert ret['cc'] in ALPHA2_WITH_LOCATION or ret['cc'] == COUNTRY_UNKNOWN, ret
            assert ret['type'] in ['asn', 'ipv4', 'ipv6'], ret
            assert ret['value'] > 0
            assert ret['status'] in ['allocated', 'assigned'], ret
            # while bgp did not exist then, there are actually some lines going
            # back all the way to the nix epoch, likely because the dates were
            # not recorded and so they use the epoch as a fallback (?)
            assert ret['date'] is None or ret['date'] >= date(1970, 1, 1), ret
            yield ret
        except:
            _LOG.error(f'Could not parse line: {line}')
            raise


@functools.lru_cache()
def get_all_asns(
        dt: t.Union[t.Literal['latest'], date] = 'latest',
        rirs: t.Set[RIR] = RIRS,
) -> t.Set[int]:
    ret: t.Set[int] = set()
    for rir in rirs:
        for assignment in _parse(_iter_delegation_file(rir, dt)):
            if assignment['type'] != 'asn': continue
            start = int(assignment['start'])
            assert assignment['value'] > 0, assignment
            for asn in range(start, start + assignment['value']):
                ret.add(asn)

    return ret


class ASStatsSingle(t.TypedDict):
    rir: RIR
    born: t.Optional[date]
    addr_count_v4_log_2: float
    addr_count_v6_log_2: float

ASStats = t.Dict[int, ASStatsSingle]

def _get_asstats(
        dt: t.Union[t.Literal['latest'], date], rirs: t.List[RIR],
) -> ASStats:
    ret: ASStats = {}

    class _ASStatsSingleTmp(t.TypedDict):
        rir: t.Optional[RIR]
        asns: t.Set[t.Tuple[int, t.Optional[date]]]
        addr_count_v4: int
        addr_count_v6: int

    for rir in rirs:
        by_id: t.Dict[str, _ASStatsSingleTmp] = defaultdict(lambda: {
            'rir': None,
            'asns': set(),
            'addr_count_v4': 0,
            'addr_count_v6': 0,
        })
        for assignment in _parse(_iter_delegation_file(rir, dt)):
            if assignment['id'] == None: continue

            el = by_id[assignment['id']]
            el['rir'] = assignment['registry']
            if assignment['type'] == 'ipv4':
                # if assignment['status'] != 'assigned': continue
                assert assignment['value'] >= 8, assignment
                el['addr_count_v4'] += assignment['value']
            elif assignment['type'] == 'ipv6':
                # if assignment['status'] != 'assigned': continue
                assert assignment['value'] >= 16, assignment
                el['addr_count_v6'] += 2 ** assignment['value']
            elif assignment['type'] == 'asn':
                assert assignment['value'] >= 1, assignment
                start = int(assignment['start'])
                for asn in range(start, start + assignment['value']):
                    el['asns'].add((asn, assignment['date']))



        for id, tmp in by_id.items():
            assert tmp['rir'] is not None
            # if len(tmp['asns']) != 0 \
            #    and tmp['addr_count_v4'] + tmp['addr_count_v6'] == 0:
            #     _LOG.warning((id, tmp))
            for asn, born in tmp['asns']:
                assert asn not in ret, \
                    f'ASes should not be delegated twice ({asn})'
                assert tmp['addr_count_v4'] == 0 or tmp['addr_count_v4'] >= 32, tmp
                ret[asn] = {
                    'rir': tmp['rir'],
                    'born': born,
                    'addr_count_v4_log_2': (
                        math.log(tmp['addr_count_v4'], 2)
                        if tmp['addr_count_v4'] != 0 else 0
                    ),
                    'addr_count_v6_log_2': (
                        math.log(tmp['addr_count_v6'], 2)
                        if tmp['addr_count_v6'] != 0 else 0
                    )
                }

    return ret

@functools.lru_cache()
def _get_rir_asstats(
        dt: t.Union[t.Literal['latest'], date],
        rirs: str,
) -> ASStats:
    _rirs = t.cast(t.List[RIR], list(sorted(rirs.split(','))))

    cache_name = f'asstats_'
    cache_name += dt.strftime("%Y%m%d") if dt != 'latest' else 'latest'
    cache_name += '_' + ('_'.join(_rirs) if len(_rirs) != len(RIRS) else 'all')

    cache = PickleFileCache(cache_name, lambda: _get_asstats(dt, _rirs))
    # cache.invalidate()

    return cache.get()


def get_rir_asstats_all(
        dt: t.Union[t.Literal['latest'], date] = 'latest',
        rirs: t.Set[RIR] = RIRS,
) -> ASStats:
    return _get_rir_asstats(dt, ','.join(rirs))


def get_rir_asstats(
        asn: int,
        dt: t.Union[t.Literal['latest'], date] = 'latest',
        rirs: t.Set[RIR] = RIRS,
) -> t.Optional[ASStatsSingle]:
    stats = _get_rir_asstats(dt, ','.join(rirs))
    return stats[asn] if asn in stats else None


if __name__ == '__main__':
    dt = date.fromisoformat('2023-05-01')
    print(get_rir_asstats(39063, dt))
    print(get_rir_asstats(3320, dt))
    print(get_rir_asstats(13335, dt))
    print(get_rir_asstats(13247, dt))

    