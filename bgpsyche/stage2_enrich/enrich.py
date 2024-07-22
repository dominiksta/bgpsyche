from datetime import date, datetime
import typing as t

from bgpsyche.service.bgp_markov_chain import (
    get_as_path_confidence, get_link_confidence,
    markov_chain_from_ripe_ris
)
from bgpsyche.service.bgp_path_snippet_length import longest_real_snippet_len
from bgpsyche.service.ext.asrank import get_asrank_customer_cone_size
from bgpsyche.service.ext.caida_asrel import get_caida_asrel
from bgpsyche.service.ext.ripe_as_names_countries import get_ripe_as_country
from bgpsyche.service.ext.rir_delegations import get_rir_asstats
from bgpsyche.stage2_enrich.as_category import get_as_category
from bgpsyche.stage2_enrich.global_trade import get_normalized_trade_relationship
from bgpsyche.stage2_enrich.democracy_index import get_democracy_index
from bgpsyche.stage2_enrich.types import ASFeaturesRaw, LinkFeatures, PathFeatures
from bgpsyche.util.bgp.valley_free import path_is_valley_free
from bgpsyche.util.geo import ALPHA2_OFFICIAL, COUNTRY_DISTANCES


def enrich_asn(asn: int, path: t.List[int]) -> ASFeaturesRaw:
    dt = datetime.fromisoformat('2023-05-01T00:00')

    country = get_ripe_as_country(asn)
    democracy_index = (
        get_democracy_index()[country] if country in get_democracy_index()
        else None
    )

    ret: ASFeaturesRaw = {
        'country_democracy_index': democracy_index,
        'distance_from_path_beginning_km': 0,
        'as_rank_cone': get_asrank_customer_cone_size(asn),
        'as_category': get_as_category(asn),
        'rirstat_addr_count_v4': None,
        'rirstat_addr_count_v6': None,
        'rirstat_born': None,
    }

    c_beg, c_asn = get_ripe_as_country(path[0]), get_ripe_as_country(asn)
    if c_beg in ALPHA2_OFFICIAL and c_asn in ALPHA2_OFFICIAL:
        ret['distance_from_path_beginning_km'] = COUNTRY_DISTANCES[c_beg][c_asn]

    rirstats = get_rir_asstats(asn, dt)
    if rirstats is not None:
        ret['rirstat_born'] = rirstats['born']
        ret['rirstat_addr_count_v4'] = rirstats['addr_count_v4_log_2']
        ret['rirstat_addr_count_v6'] = rirstats['addr_count_v6_log_2']

    return ret


def enrich_link(source: int, sink: int) -> LinkFeatures:
    dt = date.fromisoformat('2023-05-01')
    rels = get_caida_asrel(dt)
    c1, c2 = get_ripe_as_country(source), get_ripe_as_country(sink)
    counts_full, counts_per_dest = markov_chain_from_ripe_ris(dt)

    ret: LinkFeatures = {
        'rel':  rels[source][sink] if source in rels and sink in rels[source] else None,
        'confidence_per_dest': (
            get_link_confidence(sink, counts_per_dest[sink][source])
            if (sink in counts_per_dest and source in counts_per_dest[sink]) else 0
        ),
        'confidence_full': (
            get_link_confidence(sink, counts_full[source])
            if source in counts_full else 0
        ),
        'distance_km': 0,
        'trade_factor': 0,
    }

    if c1 in ALPHA2_OFFICIAL and c2 in ALPHA2_OFFICIAL:
        ret['distance_km'] = int(COUNTRY_DISTANCES[c1][c2])
        ret['trade_factor'] = \
            get_normalized_trade_relationship(c1, c2)

    return ret


def enrich_path(path: t.List[int]) -> PathFeatures:
    dt = datetime.fromisoformat('2023-05-01T00:00')
    counts_full, counts_per_dest = markov_chain_from_ripe_ris(dt)

    return {
        'length': len(path),
        'is_valley_free': path_is_valley_free(get_caida_asrel(dt), path),
        'longest_real_snippet': longest_real_snippet_len(path),
        'confidence_per_dest': (
            get_as_path_confidence(path, counts_per_dest[path[-1]])
            if path[-1] in counts_per_dest else 0
        ),
        'confidence_full': get_as_path_confidence(path, counts_full),
    }