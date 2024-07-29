from collections import defaultdict
from datetime import date
import logging
import typing as t
import math

import matplotlib.pyplot as plt
from bgpsyche.logging_config import logging_setup
from bgpsyche.research.plot_geo import plot_val_by_country
from bgpsyche.service.ext import peeringdb
from bgpsyche.service.ext.ripe_as_names_countries import get_ripe_as_country
from bgpsyche.service.ext.rir_delegations import get_rir_asstats_all
from bgpsyche.stage2_enrich.as_category import get_as_category

logging_setup()
_LOG = logging.getLogger(__name__)

def _research_peeringdb_distribution():

    pdb_asns = peeringdb.Client.get_all_asns()
    _LOG.info(f'pdb asn count: {len(pdb_asns)}')

    _pdb_types: t.Dict[int, peeringdb.NetworkType] = {
        asn: peeringdb.Client.get_network_by_asn(asn).info_type
        for asn in pdb_asns
    }
    pdb_set = {
        asn
        for asn, _pdb_type in _pdb_types.items()
        if _pdb_type is not None and _pdb_type != 'Not Disclosed'
    }
    _LOG.info(f'info_type set: {len(pdb_set)}')

    all_asns = \
        set(get_rir_asstats_all(dt=date.fromisoformat('2024-07-27')).keys())\
        .union(pdb_asns)
    _LOG.info(f'all asn count: {len(all_asns)}')

    # bgpsyche_set = {
    #     asn
    #     for asn in all_asns
    #     if get_as_category(asn) != 'Unknown'
    # }
    # _LOG.info(f'plus asdb set: {len(bgpsyche_set)}')


    assert len(pdb_asns.difference(all_asns)) == 0, pdb_asns.difference(all_asns)

    asn2iso2 = { asn: get_ripe_as_country(asn) for asn in all_asns }

    iso22asns: t.Dict[str, t.Set[int]] = defaultdict(set)
    for asn, cc in asn2iso2.items(): iso22asns[cc].add(asn)

    # plt.figure('percent')
    # data = {
    #     iso2: len(pdb_asns.intersection(iso22asns[iso2])) / len(iso22asns[iso2])
    #     for iso2 in iso22asns.keys()
    # }
    # plot_val_by_country(plt.gca(), data)

    figsize = (9, 4)

    plt.figure('percent_set', figsize=figsize)
    data = {
        iso2: (len(pdb_set.intersection(iso22asns[iso2])) / len(iso22asns[iso2]))
        for iso2 in iso22asns.keys()
    }
    plot_val_by_country(plt.gca(), data, {
        'legend': True,
    })

    plt.figure('in_pdb_log', figsize=figsize)
    data = {
        iso2: math.log(len(pdb_asns.intersection(iso22asns[iso2])) or 1, 10)
        for iso2 in iso22asns.keys()
    }
    plot_val_by_country(plt.gca(), data, {
        'legend': True,
    })

    plt.figure('all_log', figsize=figsize)
    data = {
        iso2: math.log(len(iso22asns[iso2]) or 1, 10)
        for iso2 in iso22asns.keys()
    }
    plot_val_by_country(plt.gca(), data, {
        'legend': True,
    })

    plt.show()
    







if __name__ == '__main__': _research_peeringdb_distribution()