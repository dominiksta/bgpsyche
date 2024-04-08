from datetime import date, datetime
from statistics import mean
import typing as t
from bgpsyche.service.bgp_markov_chain import get_as_path_confidence, markov_chain_from_ripe_ris
from bgpsyche.service.bgp_path_snippet_length import longest_real_snippet

from bgpsyche.service.ext.asrank import get_asrank_customer_cone_size
from bgpsyche.service.ext.caida_asrel import get_caida_asrel
from bgpsyche.service.ext.ripe_as_names_countries import get_ripe_as_country
from bgpsyche.service.ext.rir_delegations import get_rir_asstats
from bgpsyche.stage2_enrich.as_category import get_as_category
from bgpsyche.stage2_enrich.global_trade import get_trade_relationships
from bgpsyche.stage2_enrich.democracy_index import get_democracy_index
from bgpsyche.stage2_enrich.types import ASFeaturesRaw, LinkFeatures, PathFeatures
from bgpsyche.util.bgp.valley_free import path_is_valley_free
from bgpsyche.util.geo import ALPHA2_OFFICIAL, COUNTRY_DISTANCES


def enrich_asn(asn: int) -> ASFeaturesRaw:
    dt = datetime.fromisoformat('2023-05-01T00:00')

    country = get_ripe_as_country(asn)
    democracy_index = (
        get_democracy_index()[country] if country in get_democracy_index()
        else None
    )

    ret: ASFeaturesRaw = {
        'country_democracy_index': democracy_index,
        'as_rank_cone': get_asrank_customer_cone_size(asn),
        'as_category': get_as_category(asn),
        'rirstat_addr_count_v4': None,
        'rirstat_addr_count_v6': None,
        'rirstat_born': None,
    }

    rirstats = get_rir_asstats(asn, dt)
    if rirstats is not None:
        ret['rirstat_born'] = rirstats['born']
        ret['rirstat_addr_count_v4'] = rirstats['addr_count_v4_log_2']
        ret['rirstat_addr_count_v6'] = rirstats['addr_count_v6_log_2']

    return ret


def enrich_link(source: int, sink: int) -> LinkFeatures:
    rels = get_caida_asrel(date.fromisoformat('2023-05-01'))
    trade = get_trade_relationships()
    c1, c2 = get_ripe_as_country(source), get_ripe_as_country(sink)

    ret: LinkFeatures = {
        'rel':  rels[source][sink] if source in rels and sink in rels[source] else None,
        'distance_km': 0,
        'trade_service_volume_million_usd': 0,
    }

    if c1 in ALPHA2_OFFICIAL and c2 in ALPHA2_OFFICIAL:
        ret['distance_km'] = int(COUNTRY_DISTANCES[c1][c2])

        for direction in ['imports', 'exports']:
            if c1 in trade[direction] and c2 in trade[direction][c1]:
                ret['trade_service_volume_million_usd'] += trade[direction][c1][c2]

    return ret


def enrich_path(path: t.List[int]) -> PathFeatures:
    dt = datetime.fromisoformat('2023-05-01T00:00')

    return {
        'length': len(path),
        'is_valley_free': path_is_valley_free(get_caida_asrel(dt), path),
        'longest_real_snippet': len(longest_real_snippet(path)),
        'per_dest_markov_confidence': get_as_path_confidence(
            path,
            markov_chain_from_ripe_ris(dt)
        )
    }