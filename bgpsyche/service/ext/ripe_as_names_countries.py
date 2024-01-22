from pprint import pformat
import typing as t
import functools
import logging

from bgpsyche.util.const import HERE
from bgpsyche.util.net import download_file_cached
from bgpsyche.util.geo import (
    ALPHA2_OFFICIAL, ALPHA2_TRANSITIONAL, ALPHA2_EXCEPTIONALLY_RESERVED,
    Alpha2Official, Alpha2Transitional, Alpha2ExceptionallyReserved, CountryUnknown
)

_LOG = logging.getLogger(__name__)

AsnTxtCountryCode = t.Union[
    Alpha2Official, Alpha2Transitional, Alpha2ExceptionallyReserved,
    CountryUnknown,
]

class ASBasicInfo(t.TypedDict):
    name: str
    country: AsnTxtCountryCode


@functools.lru_cache()
def get_all_ripe_as_names_countries() -> t.Mapping[int, ASBasicInfo]:
    asn_name_cc_file = download_file_cached(
        'https://ftp.ripe.net/ripe/asnames/asn.txt',
        HERE / 'data' / 'asn.txt',
    )
    # we could save this all in a database, but since the file is just ~4MB,
    # keeping this in memory is likely not an issue but much fast

    count_as_unknown = { 'ZZ', '', 'UN' }

    out: t.Dict[int, ASBasicInfo] = {}
    with open(asn_name_cc_file, 'r') as f:
        for line in f:
            if line.startswith('23456 AS_TRANS'): continue
            last_comma, first_space = line.rfind(','), line.find(' ')
            country = t.cast(AsnTxtCountryCode, line[last_comma+2:-1])
            if country not in count_as_unknown \
               and country not in ALPHA2_OFFICIAL \
               and country not in ALPHA2_TRANSITIONAL \
               and country not in ALPHA2_EXCEPTIONALLY_RESERVED:
                _LOG.warning(
                    f'Weird alpha-2 code, counting as unknown: {country}'
                )
                country = 'UNKNOWN'
            if country in count_as_unknown: country = 'UNKNOWN'
            asn = int(line[:first_space])
            name = line[first_space+1:last_comma]
            out[asn] = { 'name': name, 'country': country }

    return out


def get_ripe_as_name(asn: int) -> str:
    all = get_all_ripe_as_names_countries()
    return all[asn]['name'] if asn in all else 'UNKNOWN'

def get_ripe_as_country(asn: int) -> AsnTxtCountryCode:
    all = get_all_ripe_as_names_countries()
    return all[asn]['country'] if asn in all else 'UNKNOWN'