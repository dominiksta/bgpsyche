from datetime import datetime
import typing as t

from bgpsyche.service.ext.asrank import get_asrank
from bgpsyche.service.ext.caida_asrel import get_caida_asrel
from bgpsyche.service.ext.ripe_as_names_countries import get_ripe_as_country
from bgpsyche.service.ext.rir_delegations import get_rir_asstats
from bgpsyche.util.bgp.valley_free import path_is_valley_free
from bgpsyche.stage2_enrich.types import ASFeaturesRaw, PathFeatures


def enrich_asn(asn: int) -> ASFeaturesRaw:
    dt = datetime.fromisoformat('2023-05-01T00:00')

    ret: ASFeaturesRaw = {
        'ripe_country': get_ripe_as_country(asn),
        'as_rank': get_asrank(asn),
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


def enrich_path(path: t.List[int]) -> PathFeatures:
    dt = datetime.fromisoformat('2023-05-01T00:00')

    return {
        'length': len(path),
        'is_valley_free': path_is_valley_free(get_caida_asrel(dt), path),
    }