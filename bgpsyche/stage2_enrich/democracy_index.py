from collections import defaultdict
from functools import lru_cache
import csv
from pathlib import Path
from pprint import pformat
from statistics import mean
import typing as t

from bgpsyche.util.const import HERE
from bgpsyche.util.geo import (
    ALPHA3_OFFICIAL, ALPHA3_TO_ALPHA2, Alpha2Official, Alpha3Official
)

# download manually from: https://ourworldindata.org/grapher/democracy-index-eiu?time=latest
# manual download necessary because it is a dynamically created blob from js

# country -> [0,10]
DemocracyIndex = t.Dict[Alpha2Official, float]

_DATA_FILE = Path(__file__).parent / 'democracy_index.csv'

class _DemocaryIndexFileLine(t.TypedDict):
    Entity: str
    Code: Alpha3Official
    Year: str
    democracy_eiu: str

@lru_cache()
def get_democracy_index(
        year: t.Union[t.Literal['max'], int] = 'max'
) -> DemocracyIndex:
    di: t.Dict[int, DemocracyIndex] = defaultdict(dict)

    with open(_DATA_FILE) as f:
        reader = t.cast(t.Iterable[_DemocaryIndexFileLine], csv.DictReader(f))
        for line in reader:
            if line['Code'] in ['', 'OWID_WRL']: continue
            assert line['Code'] in ALPHA3_OFFICIAL, pformat(line)
            di[int(line['Year'])][ALPHA3_TO_ALPHA2[line['Code']]] = \
                float(line['democracy_eiu'])

    if year == 'max': year = max(di.keys())
    return di[year]


@lru_cache()
def democracy_index_avg() -> float:
    return mean(get_democracy_index().values())


if __name__ == '__main__':
    print(pformat(get_democracy_index()))
