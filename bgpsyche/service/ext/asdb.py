from collections import defaultdict
import logging
import typing as t
import csv
from functools import lru_cache
from pathlib import Path

from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.util.const import DATA_DIR
from bgpsyche.util.net import download_file_cached

logging_setup()
_LOG = logging.getLogger(__name__)
_DATA_DIR = DATA_DIR / 'asdb'

NAICSLiteSelection = t.Literal[

    # Layer 1
    # ----------------------------------------------------------------------
    'Media, Publishing, and Broadcasting',
    'Finance and Insurance',
    'Education and Research',
    'Service',
    'Agriculture, Mining, and Refineries (Farming, Greenhouses, Mining, Forestry, and Animal Farming)',
    'Community Groups and Nonprofits',
    'Construction and Real Estate',
    'Museums, Libraries, and Entertainment',
    'Utilities (Excluding Internet Service)',
    'Health Care Services',
    'Travel and Accommodation',
    'Freight, Shipment, and Postal Services',
    'Government and Public Administration',
    'Retail Stores, Wholesale, and E-commerce Sites',
    'Manufacturing',
    'Other',
    'Unknown',

    # Computer and Technology
    # ----------------------------------------------------------------------

    'Computer and IT - Internet Service Provider (ISP)',
    'Computer and IT - Phone Provider',
    'Computer and IT - Hosting, Cloud Provider, Data Center, Server Colocation',
    'Computer and IT - Computer and Network Security',
    'Computer and IT - Software Development',
    'Computer and IT - Technology Consulting Services',
    'Computer and IT - Satellite Communication',
    'Computer and IT - Search',
    'Computer and IT - Internet Exchange Point (IXP)',
    'Computer and IT - Other',
    'Computer and IT - Unknown',
    # we assign 'Computer and IT - <Empty String>' to 'Computer and IT - Unknown'
]

NAICSLITE_SELECTION: t.Set[NAICSLiteSelection] = set(t.get_args(NAICSLiteSelection))

def _download_asdb(year = 2024, month = 1) -> Path:
    _DATA_DIR.mkdir(exist_ok=True, parents=True)
    fname = f'{year}-{month:0>2}_categorized_ases.csv'
    return download_file_cached(
        f'https://asdb.stanford.edu/data/{fname}',
        _DATA_DIR / 'asdb' / fname
    )


ASdb = t.Dict[int, NAICSLiteSelection]

@lru_cache()
def get_asdb_full(year = 2024, month = 1) -> ASdb:

    def _get() -> ASdb:
        csv_file = _download_asdb(year, month)
        asdb: ASdb = {}

        _LOG.info('Parsing ASdb file...')

        def _insert(cat: str):
            assert cat in NAICSLITE_SELECTION, { 'cat': cat, 'line': line }
            asdb[asn] = cat

        computer_l2_fix_map = {
            '': 'Unknown',
            # yes yes very data sanitization very good yes yes
            'Hosting and Cloud Provider': \
            'Hosting, Cloud Provider, Data Center, Server Colocation',
            'Hosting and Cloud Provicer': \
            'Hosting, Cloud Provider, Data Center, Server Colocation',
        }

        def _insert_computer_l2(cat_l2: str):
            cat = computer_l2_fix_map[cat_l2] \
                if cat_l2 in computer_l2_fix_map else cat_l2
            cat = 'Computer and IT - ' + cat
            _insert(cat)


        with open(csv_file, 'r') as f:
            for line in csv.reader(f, delimiter=',', quotechar='"'):
                if len(line) == 0: continue
                if line[0].startswith('ASN'): continue # header line

                if len([
                        col for col in line if col.startswith('AS')
                ]) != 1:
                    _LOG.warning(f'Ignoring weird asdb line: {line}')
                    continue

                asn = int(line[0].split('AS')[1])
                assert asn not in asdb

                cats    = [
                    (line[i], line[i+1]) for i in range(1, len(line), 2)
                    if line[i] != ''
                ]
                cats_l1 = [ cat[0] for cat in cats ]
                assert len(cats_l1) >= 1

                # if the first l1 category is not computer&it, we take that. but
                # if any other l1 category is computer&l1, we prioritize that
                # over anything else

                if 'Computer and Information Technology' in cats_l1:

                    if 'Computer and Information Technology' != cats_l1[0]:
                        _insert(cats_l1[0])
                        continue

                    computer_l2 = [
                        cat[1] for cat in cats
                        if cat[0] == 'Computer and Information Technology'
                    ]

                    assert len(computer_l2) >= 0

                    if len(computer_l2) == 1:
                        _insert_computer_l2(computer_l2[0])
                    else:
                        no_other = [ c for c in computer_l2 if c != 'Other' and c != '' ]
                        _insert_computer_l2(no_other[0] if len(no_other) != 0 else 'Other')

                else:
                    _insert(cats_l1[0])

        return dict(asdb)

    cache = PickleFileCache(f'asdb_{year}_{month:0>2}', _get)
    # cache.invalidate()
    return cache.get()


def get_asdb_primary(
        asn: int, year = 2024, month = 1
) -> NAICSLiteSelection:
    full = get_asdb_full(year, month)
    return full[asn] if asn in full else 'Unknown'


if __name__ == '__main__':
    show = {
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
    for asn, name in show.items():
        print(f'{asn: >6} ({name: >40}): {get_asdb_primary(asn)}')