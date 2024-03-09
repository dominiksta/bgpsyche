import functools
import typing as t
import bz2
from collections import defaultdict
from datetime import date
import logging
from pathlib import Path

from bgpsyche.util.bgp.relationship import (
    Relationship, RelationshipKind, Source2Sink2Rel, relationship_reverse
)
from bgpsyche.util.bgp.tier1 import TIER1_SIBLINGS_FLAT
from bgpsyche.util.const import DATA_DIR
from bgpsyche.util.net import download_file_cached

_LOG = logging.getLogger(__name__)


def download_caida_asrel(
        date: date,
        ver: t.Literal[1, 2] = 2,
) -> Path:
    fdate = date.strftime("%Y%m%d")
    file = f'{fdate}.as-rel' + ('2' if ver == 2 else '') + '.txt.bz2'
    return download_file_cached(
        f'http://data.caida.org/datasets/as-relationships/serial-{ver}/{file}',
        DATA_DIR / file
    )


_num2relationship: t.Dict[int, RelationshipKind] = { 1: 'c2p', 0: 'p2p', -1: 'p2c' }


@functools.lru_cache()
def get_caida_asrel(
        date: date,
        enforce_t1 = True,
        # - Version 1 is inferred from bgp paths using asrank (Luckie et al. "AS
        #   relationships, customer cones, and validation",
        #   10.1145/2504730.2504735)
        # - Version 2 adds ~250k (roughly double) p2p relationships inferred
        #   from bgp community attributes (Giotsas et al. "Inferring
        #   multilateral peering", 10.1145/2535372.2535390)
        ver: t.Literal[1, 2] = 2,
) -> Source2Sink2Rel:
    s2s2r: Source2Sink2Rel = defaultdict(dict)
    _LOG.info('Parsing CAIDA AS relationship file')

    def parse_relationship_line(line: str) -> Relationship:
        # <provider-as>|<customer-as>|-1
        # <peer-as>|<peer-as>|0
        split  = line.strip().split("|")
        source = int(split[ 0])
        sink   = int(split[1])
        rel    = _num2relationship[int(split[2])]
        return source, sink, rel

    asrel_file = download_caida_asrel(date, ver)
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