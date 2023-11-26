import typing as t
import functools

from bgpsyche.util.const import HERE
from bgpsyche.util.net import download_file_cached

CountryUnknown = t.Literal['UNKNOWN']

CountryName = t.Literal[
    # "normal" valid alpha 2 country codes
    # ----------------------------------------------------------------------

    'AD', 'AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AN', 'AO', 'AQ', 'AR',
    'AS', 'AT', 'AU', 'AW', 'AX', 'AZ', 'BA', 'BB', 'BD', 'BE', 'BF',
    'BG', 'BH', 'BI', 'BJ', 'BL', 'BM', 'BN', 'BO', 'BQ', 'BR', 'BS',
    'BT', 'BV', 'BW', 'BY', 'BZ', 'CA', 'CC', 'CD', 'CF', 'CG', 'CH',
    'CI', 'CK', 'CL', 'CM', 'CN', 'CO', 'CR', 'CU', 'CV', 'CW', 'CX',
    'CY', 'CZ', 'DE', 'DJ', 'DK', 'DM', 'DO', 'DZ', 'EC', 'EE', 'EG',
    'EH', 'ER', 'ES', 'ET', 'FI', 'FJ', 'FK', 'FM', 'FO', 'FR', 'GA',
    'GB', 'GD', 'GE', 'GF', 'GG', 'GH', 'GI', 'GL', 'GM', 'GN', 'GP',
    'GQ', 'GR', 'GS', 'GT', 'GU', 'GW', 'GY', 'HK', 'HM', 'HN', 'HR',
    'HT', 'HU', 'ID', 'IE', 'IL', 'IM', 'IN', 'IO', 'IQ', 'IR', 'IS',
    'IT', 'JE', 'JM', 'JO', 'JP', 'KE', 'KG', 'KH', 'KI', 'KM', 'KN',
    'KP', 'KR', 'KW', 'KY', 'KZ', 'LA', 'LB', 'LC', 'LI', 'LK', 'LR',
    'LS', 'LT', 'LU', 'LV', 'LY', 'MA', 'MC', 'MD', 'ME', 'MF', 'MG',
    'MH', 'MK', 'ML', 'MM', 'MN', 'MO', 'MP', 'MQ', 'MR', 'MS', 'MT',
    'MU', 'MV', 'MW', 'MX', 'MY', 'MZ', 'NA', 'NC', 'NE', 'NF', 'NG',
    'NI', 'NL', 'NO', 'NP', 'NR', 'NU', 'NZ', 'OM', 'PA', 'PE', 'PF',
    'PG', 'PH', 'PK', 'PL', 'PM', 'PN', 'PR', 'PS', 'PT', 'PW', 'PY',
    'QA', 'RE', 'RO', 'RS', 'RU', 'RW', 'SA', 'SB', 'SC', 'SD', 'SE',
    'SG', 'SH', 'SI', 'SJ', 'SK', 'SL', 'SM', 'SN', 'SO', 'SR', 'SS',
    'ST', 'SV', 'SX', 'SY', 'SZ', 'TC', 'TD', 'TF', 'TG', 'TH', 'TJ',
    'TK', 'TL', 'TM', 'TN', 'TO', 'TP', 'TR', 'TT', 'TV', 'TW', 'TZ',
    'UA', 'UG', 'UK', 'UM', 'US', 'UY', 'UZ', 'VA', 'VC', 'VE', 'VG',
    'VI', 'VN', 'VU', 'WF', 'WS', 'YE', 'YT', 'YU', 'ZA', 'ZM', 'ZW',

    # "special" codes
    # ----------------------------------------------------------------------

    'UNKNOWN' , # this is our own custom marker
    'ZZ'      , # 'Private Use AS' or 'Reserved AS'
    'EU'      , # 'European Union'. While this is not a country, it is a valid
                # alpha 2 code according to wikipedia:
                # https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2#Decoding_table
                # It appears to be used by ripe.
]

COUNTRIES: t.Set[str] = set(t.get_args(CountryName))


class ASBasicInfo(t.TypedDict):
    name: str
    country: CountryName


@functools.lru_cache()
def get_all_asn_basic_info() -> t.Mapping[int, ASBasicInfo]:
    asn_name_cc_file = download_file_cached(
        'https://ftp.ripe.net/ripe/asnames/asn.txt',
        HERE / 'data' / 'asn.txt',
    )
    # we could save this all in a database, but since the file is just ~4MB,
    # keeping this in memory is likely not an issue but much fast

    count_as_unknown = {
        'AP', # African Regional Intellectual Property Organization,
        'ZZ',
    }

    out: t.Dict[int, ASBasicInfo] = {}
    with open(asn_name_cc_file, 'r') as f:
        for line in f:
            if line.startswith('23456 AS_TRANS'): continue
            last_comma, first_space = line.rfind(','), line.find(' ')
            country = t.cast(CountryName, line[last_comma+2:-1])
            if country in count_as_unknown: country = 'UNKNOWN'
            assert country in COUNTRIES, f'{country} not in {COUNTRIES}'
            asn = int(line[:first_space])
            name = line[first_space+1:last_comma]
            out[asn] = { 'name': name, 'country': country }

    return out


def get_asn_basic_info(asn: int) -> ASBasicInfo:
    all = get_all_asn_basic_info()
    return all[asn] if asn in all \
        else { 'name': 'UNKNOWN', 'country': 'UNKNOWN' }