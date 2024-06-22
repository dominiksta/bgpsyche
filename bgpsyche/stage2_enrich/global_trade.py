from collections import defaultdict
from datetime import datetime
import zipfile
from functools import lru_cache
import logging
import json
from statistics import mean
import typing as t
from pathlib import Path
import csv
from pprint import pformat

from bgpsyche.util.const import DATA_DIR, HERE
from bgpsyche.util.net import download_file_cached
from bgpsyche.logging_config import logging_setup
from bgpsyche.util.geo import (
    ALPHA3_OFFICIAL, ALPHA3_TO_ALPHA2, Alpha2Official, Alpha3Official
)

logging_setup()
_LOG = logging.getLogger(__name__)

# https://data-explorer.oecd.org/vis?fs%5B0%5D=Topic%2C1%7CEconomy%23ECO%23%7CBalance%20of%20payments%23ECO_BAL%23&pg=0&fc=Topic&bp=true&snb=3&df%5Bds%5D=dsDisseminateFinalDMZ&df%5Bid%5D=DSD_BOP%40DF_BOP&df%5Bag%5D=OECD.SDD.TPS&df%5Bvs%5D=1.0&pd=%2C&dq=.....Q..&ly%5Brw%5D=MEASURE&ly%5Bcl%5D=TIME_PERIOD&ly%5Brs%5D=ACCOUNTING_ENTRY&to%5BTIME_PERIOD%5D=false

class _ServicesAnnualDatasetLine(t.TypedDict):
    IndicatorCategory: t.Literal['Trade in Commercial Services']
    IndicatorCode: t.Literal[
        'ITS_CS_AM5', # Commercial services imports by sector - annual (1980-2013)
        'ITS_CS_AM6', # Commercial services imports by sector and partner - annual
        'ITS_CS_AX5', # Commercial services exports by sector - annual  (1980-2013)
        'ITS_CS_AX6', # Commercial services exports by sector and partner - annual
    ]
    Indicator: str
    ReporterCode: str
    ReporterISO3A: Alpha3Official
    Reporter: str
    PartnerCode: str
    PartnerISO3A: Alpha3Official
    Partner: str
    ProductClassificationCode: t.Literal[
        'BOP5', # Services - Extended Balance of Payments Classification (EBOPS 2002)
        'BOP6', # Services - Extended Balance of Payments Classification (EBOPS 2010)
    ]
    ProductClassification: str
    ProductCode: str
    Product: str
    PeriodCode: t.Union[str, t.Literal[
        'S',    # Memo item: Total services
        'S200', # Memo item: Total services
        # ...
    ]]
    Period: t.Literal['Annual']
    FrequencyCode: t.Literal['A']
    Frequency: t.Literal['Annual']
    UnitCode: t.Literal['USM']
    Unit: t.Literal['Million US dollar']
    Year: str
    ValueFlagCode: t.Optional[t.Literal[
        'B', # Break in series
        'E', # Estimate
        'C', # Coverage differs
    ]]
    ValueFlag: str
    Value: str


# Unit: Million USD
CountryTradeMapping = t.Dict[Alpha2Official, t.Dict[Alpha2Official, int]]

class GlobalServicesTrade(t.TypedDict):
    imports: CountryTradeMapping
    imports_total: t.Dict[Alpha2Official, int]
    exports: CountryTradeMapping
    exports_total: t.Dict[Alpha2Official, int]


def _parse_wto_services_annual_dataset(path: Path) -> GlobalServicesTrade:
    reported_imports: CountryTradeMapping = defaultdict(dict)
    reported_exports: CountryTradeMapping = defaultdict(dict)

    ret: GlobalServicesTrade = {
        'imports': defaultdict(dict), 'imports_total': {},
        'exports': defaultdict(dict), 'exports_total': {},
    }

    skip_codes: t.Set[str] = {
        '000', # World
        'EUT', # Extra EU Trade
        '918', # European Union
        '928', # European Union (28) ?
        '946', # Group of Seven
        '910', # Commonwealth of Independent States (CIS)
        '950', # Africa
        '945', # OPEC
        '931', # North America
        '970', # Asia
        '535', # Bonaire, Sint Eustatius and Saba
        '975', # Association of Southeast Asian Nations (ASEAN) (lol)
        '474', # Martinique
        '254', # French Guinea
        '638', # Reunion
        '965', # Middle East
        '905', # Europe
    }

    # tmp
    skip_alpha3: t.Set[str] = {
        'PYF', # French Polynesia
        'SCG', # Serbia and Montenegro
        'CHT', # Chinese Taipei
        'ANT', # Netherland Antilles
    }

    alpha3_fix: t.Dict[str, Alpha3Official] = {
        'ROM': 'ROU',
        'BLX': 'BEL', # "Belgium-Luxembourg -> Belgium"
    }

    with open(path, encoding='latin-1') as f:
        reader = t.cast(t.Iterable[_ServicesAnnualDatasetLine], csv.DictReader(f))
        for i, line in enumerate(reader):
            if i % 100_000 == 0: _LOG.info(f'Line {i}')

            if line['PartnerCode'] in skip_codes or \
               line['ReporterCode'] in skip_codes: continue

            if line['PartnerISO3A'] in skip_alpha3 or \
               line['ReporterISO3A'] in skip_alpha3: continue

            if (line['PartnerISO3A'] in alpha3_fix):
                line['PartnerISO3A'] = alpha3_fix[line['PartnerISO3A']]
            if (line['ReporterISO3A'] in alpha3_fix):
                line['ReporterISO3A'] = alpha3_fix[line['ReporterISO3A']]

            if (line['PartnerCode'] == '312'): line['PartnerISO3A'] = 'GLP'
            if (line['ReporterCode'] == '312'): line['ReporterISO3A'] = 'GLP'


            if line['PartnerISO3A'] == '':
                _LOG.warning(f'Non-alpha3 partner {line["Partner"]}, skipping')
                continue

            if line['ReporterISO3A'] == '':
                _LOG.warning(f'Non-alpha3 reporter {line["Reporter"]}, skipping')
                continue

            source, sink = line['ReporterISO3A'], line['PartnerISO3A']
            assert source in ALPHA3_OFFICIAL and sink in ALPHA3_OFFICIAL, pformat(line)
            if line['IndicatorCode'] in ['ITS_CS_AX6', 'ITS_CS_AX5']:
                reported_imports[ALPHA3_TO_ALPHA2[source]][ALPHA3_TO_ALPHA2[sink]] = \
                    int(line['Value'])
            elif line['IndicatorCode'] in ['ITS_CS_AM6', 'ITS_CS_AM5']:
                reported_exports[ALPHA3_TO_ALPHA2[source]][ALPHA3_TO_ALPHA2[sink]] = \
                    int(line['Value'])

    # countries can disagree on their reported exports/imports. example: country
    # A says it has exported 100M USD to country B, but country B says it has
    # only import 85M USD from country A. as a basic approximation, we take the
    # mean value of the two reported values here.
    # ----------------------------------------------------------------------

    for source, sink2val in reported_imports.items():
        for sink, val in sink2val.items():
            if sink in reported_exports and source in reported_exports[sink]:
                ret['imports'][source][sink] = \
                    round(mean([reported_exports[sink][source], val]))
            else:
                ret['imports'][source][sink] = val

    for source, sink2val in reported_exports.items():
        for sink, val in sink2val.items():
            if sink in reported_imports and source in reported_imports[sink]:
                ret['exports'][source][sink] = \
                    round(mean([reported_imports[sink][source], val]))
            else:
                ret['exports'][source][sink] = val

    # compute totals
    # ----------------------------------------------------------------------

    ret['exports_total'] = {
        country: sum(partners.values())
        for country, partners in ret['exports'].items()
    }

    ret['imports_total'] = {
        country: sum(partners.values())
        for country, partners in ret['imports'].items()
    }
            
    return {
        **ret,
        'exports': dict(ret['exports']), 'imports': dict(ret['imports'])
    }


def _download_wto_services_annual_dataset():
    # see https://www.wto.org/english/res_e/statis_e/trade_datasets_e.htm
    year = datetime.now().year
    base_dir = DATA_DIR / 'wto' / f'{year}'
    file = download_file_cached(
        'https://stats.wto.org/assets/UserGuide/services_annual_dataset.zip',
        base_dir / 'services_annual_dataset.zip'
    )
    extracted = base_dir / 'services_annual_dataset.csv'
    if not extracted.exists():
        _LOG.info('extracting...')
        with zipfile.ZipFile(file, 'r') as z: z.extractall(base_dir)
    return extracted


_DATA_FILE = Path(__file__).parent / 'global_trade.json'


def _update_wto_services_annual():
    trade = _parse_wto_services_annual_dataset(
        _download_wto_services_annual_dataset()
    )
    with open(_DATA_FILE, 'w') as f: f.write(json.dumps(trade, indent=2))


@lru_cache()
def get_trade_relationships() -> GlobalServicesTrade:
    with open(_DATA_FILE) as f: return json.loads(f.read())


def get_normalized_trade_relationship(
        c1: Alpha2Official,
        c2: Alpha2Official,
) -> float: # [0;1]
    trade = get_trade_relationships()
    ret = 0.
    for direction in ['imports', 'exports']:
        if c1 in trade[direction] and c2 in trade[direction][c1]:
            assert c1 in trade[f'{direction}_total']
            if trade[f'{direction}_total'][c1] == 0: continue
            # HACK: no idea why this would be negative, but its only a small
            # handful of edge-cases
            if trade[direction][c1][c2] < 0: continue
            ret += round(
                trade[direction][c1][c2] / trade[f'{direction}_total'][c1], 2
            )

    ret = ret / 2
    assert ret >= 0 and ret <= 1, { 'ret': ret, 'c1': c1, 'c2': c2 }
    return ret


if __name__ == '__main__': _update_wto_services_annual()