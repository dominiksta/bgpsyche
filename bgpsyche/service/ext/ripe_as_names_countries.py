from pprint import pformat
import typing as t
import functools

from bgpsyche.util.const import HERE
from bgpsyche.util.net import download_file_cached
from bgpsyche.util.geo import CountryName, COUNTRIES

class ASBasicInfo(t.TypedDict):
    name: str
    country: CountryName


@functools.lru_cache()
def get_all_ripe_as_names_countries() -> t.Mapping[int, ASBasicInfo]:
    asn_name_cc_file = download_file_cached(
        'https://ftp.ripe.net/ripe/asnames/asn.txt',
        HERE / 'data' / 'asn.txt',
    )
    # we could save this all in a database, but since the file is just ~4MB,
    # keeping this in memory is likely not an issue but much fast

    count_as_unknown = {
        'AP', # African Regional Intellectual Property Organization,
        'ZZ',
        '',
    }

    out: t.Dict[int, ASBasicInfo] = {}
    with open(asn_name_cc_file, 'r') as f:
        for line in f:
            if line.startswith('23456 AS_TRANS'): continue
            last_comma, first_space = line.rfind(','), line.find(' ')
            country = t.cast(CountryName, line[last_comma+2:-1])
            if country in count_as_unknown: country = 'UNKNOWN'
            asn = int(line[:first_space])
            name = line[first_space+1:last_comma]
            if name == '': name = 'UNKNOWN'
            out[asn] = { 'name': name, 'country': country }
            assert country in COUNTRIES, pformat({
                'country': country, 'name': name, 'line': line,
            })

    return out


def get_ripe_as_name(asn: int) -> str:
    all = get_all_ripe_as_names_countries()
    return all[asn]['name'] if asn in all else 'UNKNOWN'

def get_ripe_as_country(asn: int) -> CountryName:
    all = get_all_ripe_as_names_countries()
    return all[asn]['country'] if asn in all else 'UNKNOWN'