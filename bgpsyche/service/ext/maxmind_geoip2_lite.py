import functools
from pprint import pformat
import typing as t
from ipaddress import IPv4Network, IPv6Network, ip_address
from zipfile import ZipFile
import csv
import logging

from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.util.const import DATA_DIR, ENV
from bgpsyche.util.geo import ALPHA2_WITH_LOCATION, Alpha2WithLocation
from bgpsyche.util.net.download import download_file_cached
from bgpsyche.util.net import NetworkPrefixTreeV4, NetworkPrefixTreeV6
from bgpsyche.util.net.typ import IPAddress, IPNetwork

_LOG = logging.getLogger(__name__)


def _download_geoip2_lite():
    # see the "download files" section after creating a free account
    url = f'https://download.maxmind.com/app/geoip_download' + \
        f'?edition_id=GeoLite2-Country-CSV' + \
        f'&license_key={ENV["maxmind_geoip2_lite"]["license_key"]}' + \
        f'&suffix=zip'

    return download_file_cached(url, DATA_DIR / 'maxmin_geoip2_lite.zip')


class GeoIP2(t.TypedDict):
    net2country: t.Dict[IPNetwork, Alpha2WithLocation]
    networks_v4: NetworkPrefixTreeV4
    networks_v6: NetworkPrefixTreeV6


def _parse_geoip2_lite() -> GeoIP2:
    net2id: t.Dict[IPNetwork, int] = {}
    id2iso: t.Dict[int, Alpha2WithLocation] = {}
    tree_v4 = NetworkPrefixTreeV4()
    tree_v6 = NetworkPrefixTreeV6()

    ignore = {
        '',
        '6255147' # Asia
    }

    with ZipFile(_download_geoip2_lite()) as zip:
        files = zip.namelist()

        file_id2iso = [
            f for f in files if 'GeoLite2-Country-Locations-en.csv' in f
        ][0]
        with zip.open(file_id2iso) as f:
            reader = csv.DictReader(
                f.read().decode('UTF-8').split('\n'),
                delimiter=','
            )
            for row in reader:
                country = row['country_iso_code']
                if row['geoname_id'] == '6255148': country = 'EU'
                if country == '': continue # skip continents
                assert country in ALPHA2_WITH_LOCATION, pformat({
                    'country': country,
                    'row': row
                })
                id2iso[int(row['geoname_id'])] = t.cast(Alpha2WithLocation, country)

        file_v42id = [
            f for f in files if 'GeoLite2-Country-Blocks-IPv4.csv' in f
        ][0]
        with zip.open(file_v42id) as f:
            reader = csv.DictReader(
                f.read().decode('UTF-8').split('\n'),
                delimiter=','
            )
            iter = 0
            for row in reader:
                iter += 1
                if iter % 100_000 == 0:
                    _LOG.info(
                        f'Parsing MaxMind GeoIP2 Lite v4: {iter} lines'
                    )
                geo_col = row['geoname_id']
                # if geo_col in ignore: print(pformat(row))
                if geo_col in ignore: geo_col = row['registered_country_geoname_id']
                if geo_col in ignore: geo_col = row['represented_country_geoname_id']
                if geo_col in ignore:
                    _LOG.warning(f'Found entry with no country: {row}')
                    continue
                # print(pformat(row))
                net = IPv4Network(row['network'])
                tree_v4.insert(net)
                net2id[net] = int(geo_col)

        file_v62id = [
            f for f in files if 'GeoLite2-Country-Blocks-IPv6.csv' in f
        ][0]
        with zip.open(file_v62id) as f:
            reader = csv.DictReader(
                f.read().decode('UTF-8').split('\n'),
                delimiter=','
            )
            iter = 0
            for row in reader:
                iter += 1
                if iter % 100_000 == 0:
                    _LOG.info(
                        f'Parsing MaxMind GeoIP2 Lite v6: {iter} lines'
                    )
                geo_col = row['geoname_id']
                if geo_col in ignore: geo_col = row['registered_country_geoname_id']
                if geo_col in ignore: geo_col = row['represented_country_geoname_id']
                if geo_col in ignore: continue
                net = IPv6Network(row['network'])
                tree_v6.insert(net)
                net2id[net] = int(geo_col)

        
    net2country: t.Dict[IPNetwork, Alpha2WithLocation] = {}
    id_not_found: t.Set[int] = set()
    for net, id in net2id.items():
        if id not in id2iso:
            id_not_found.add(id)
            continue
        net2country[net] = id2iso[id]

    if len(id_not_found) != 0:
        _LOG.warning(f'Could not find countries for ids: {id_not_found}')

    return {
        'net2country': net2country,
        'networks_v4': tree_v4,
        'networks_v6': tree_v6,
    }


@functools.lru_cache()
def _cached_geoip2_lite() -> GeoIP2:
    cache = PickleFileCache(
        'geoip2_lite',
        _parse_geoip2_lite,
    )
    # cache.invalidate()
    ret = cache.get()
    _LOG.info('Loaded GeoIP2 DB')
    return ret


@functools.lru_cache(maxsize=10_000)
def geoip2_find_ip(ip: IPAddress) -> t.Optional[Alpha2WithLocation]:
    db = _cached_geoip2_lite()
    net = db['networks_v4'].search(ip) \
        if ip.version == 4 \
        else db['networks_v6'].search(ip)
    if net is None: return None
    if net in db['net2country']: return db['net2country'][net]
    return None


@functools.lru_cache(maxsize=10_000)
def geoip2_find_network(net: IPNetwork) -> t.Optional[Alpha2WithLocation]:
    return geoip2_find_ip(net.broadcast_address)
    

if __name__ == '__main__':
    # _parse_geoip2_lite()

    print(geoip2_find_ip(ip_address('194.145.125.5')))
    print(geoip2_find_ip(ip_address('8.8.8.8')))
    print(geoip2_find_ip(ip_address('142.250.186.110')))
    print(geoip2_find_ip(ip_address('80.158.67.40')))
    # print()