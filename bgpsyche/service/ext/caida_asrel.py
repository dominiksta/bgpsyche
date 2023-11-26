import typing as t
import bz2
from collections import defaultdict
from datetime import datetime
import logging
from pathlib import Path

from bgpsyche.util.bgp.relationship import (
    Relationship, RelationshipKind, Source2Sink2Rel, relationship_reverse
)
from bgpsyche.util.bgp.tier1 import TIER1_SIBLINGS_FLAT
from bgpsyche.util.const import DATA_DIR
from bgpsyche.util.net import download_file_cached

_LOG = logging.getLogger(__name__)


def download_caida_asrel(date: datetime) -> Path:
    file = f'{date.strftime("%Y%m%d")}.as-rel.txt.bz2'
    # TODO: difference between serial-1 and serial-2 ?
    return download_file_cached(
        f'http://data.caida.org/datasets/as-relationships/serial-1/{file}',
        DATA_DIR / file
    )


_num2relationship: t.Dict[int, RelationshipKind] = { 1: 'c2p', 0: 'p2p', -1: 'p2c' }


def read_caida_file(
        date: datetime,
        enforce_t1 = True,
) -> Source2Sink2Rel:
    s2s2r: Source2Sink2Rel = defaultdict(dict)
    _LOG.info('Parsing CAIDA AS relationship file')

    def parse_relationship_line(line: str) -> Relationship:
        # <provider-as>|<customer-as>|-1
        # <peer-as>|<peer-as>|0
        source, sink, rel = line.strip().split("|")
        return int(source), int(sink), _num2relationship[int(rel)]

    asrel_file = download_caida_asrel(date)
    with bz2.open(asrel_file, 'rt') as f:
        asns: t.Set[int] = set()
        count_parsed = 0
        for line in f:
            count_parsed += 1
            if line[0] == '#': continue # skip comments and metadata
            source, sink, rel = parse_relationship_line(line)
            asns.add(source)
            asns.add(sink)

            if enforce_t1:
                source_t1, sink_t1 = \
                    source in TIER1_SIBLINGS_FLAT, sink in TIER1_SIBLINGS_FLAT
                if       source_t1 and not sink_t1: rel = 'p2c'
                elif not source_t1 and     sink_t1: rel = 'c2p'
                elif     source_t1 and     sink_t1: rel = 'p2p'
            s2s2r[source][sink] = rel
            # adding inverted relationships here should not be necessary since
            # there should not be any duplicates in the caida dataset, however
            # we still do it just to not do things differently in different
            # parts of the code.
            s2s2r[sink][source] = relationship_reverse(rel)

    return s2s2r