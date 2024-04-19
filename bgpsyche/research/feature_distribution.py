import typing as t
import logging
import math

import matplotlib.pyplot as plt
from tqdm import tqdm
from bgpsyche.logging_config import logging_setup
from bgpsyche.service.ext import peeringdb
from bgpsyche.service.ext.asdb import NAICSLITE_SELECTION, get_asdb_primary
from bgpsyche.service.ext.asrank import get_asrank_customer_cone_sizes_full
from bgpsyche.service.ext.ripe_as_names_countries import get_ripe_as_country
from bgpsyche.service.ext.rir_delegations import get_rir_asstats_all
from bgpsyche.stage2_enrich.as_category import AS_CATEGORY, get_as_category
from bgpsyche.stage2_enrich.democracy_index import get_democracy_index
from bgpsyche.stage3_rank.make_dataset import make_dataset
from bgpsyche.stage3_rank.vectorize_features import AS_FEATURE_VECTOR_NAMES
from bgpsyche.util.cancel_iter import cancel_iter

logging_setup()
_LOG = logging.getLogger(__name__)

def _plot_customer_cone():
    data = [
        math.log(cone or 1)
        for cone in get_asrank_customer_cone_sizes_full().values()
    ]
    
    plt.ecdf(data)
    plt.show()


def _plot_category_distribution():

    pdb_asns = peeringdb.Client.get_all_asns()
    all_asns = set(get_rir_asstats_all().keys()).union(pdb_asns)

    # all_asns = { asn for el in tqdm(make_dataset()) for asn in el['path'] }

    
    # --- bgpsyche ---
    # get_cat = get_as_category
    # labels = list(AS_CATEGORY)
    # labels_rename = {}
    # figsize = (5, 4)

    # --- asdb: bgpsyche selection --- 
    # get_cat = get_asdb_primary
    # labels = list(NAICSLITE_SELECTION)
    # labels_rename = {
    #     'Agriculture, Mining, and Refineries (Farming, Greenhouses, Mining, Forestry, and Animal Farming)': 'Agriculture, Mining, and Refineries',
    #     'Computer and IT - Internet Service Provider (ISP)': 'Computer and IT - ISP',
    #     'Computer and IT - Internet Exchange Point (IXP)': 'Computer and IT - IXP',
    #     'Computer and IT - Hosting, Cloud Provider, Data Center, Server Colocation': 'Computer and IT - Hosting, Cloud Provider, [...]',
    # }
    # figsize = (7, 6)

    # --- peeringdb --- 
    def get_cat(asn):
        pdb = peeringdb.Client.get_network_by_asn(asn)
        return pdb.info_type if pdb else None
    labels = list(t.get_args(peeringdb.NetworkType))
    labels_rename = {}
    figsize = (5, 4)


    cats = [ get_cat(asn) for asn in cancel_iter(tqdm(all_asns)) ]
    cats = [ cat for cat in cats if cat is not None ]
    data = { cat: (cats.count(cat) / len(cats)) * 100 for cat in labels }

    labels = sorted(data.keys(), key=lambda cat: -data[cat])
    labels = [
        labels_rename[l] if l in labels_rename else l
        for l in labels
    ]

    y = sorted(data.values(), reverse=True)

    plt.figure(1, figsize=figsize)
    plt.barh(labels, y)
    # plt.setp(plt.gca().get_xticklabels(), rotation=90, ha='right')
    plt.xlabel('%')
    plt.tight_layout()
    plt.show()


def _plot_democracy_index():
    all_asns = set(get_rir_asstats_all().keys())

    di = get_democracy_index()
    data = [
        di[get_ripe_as_country(asn)] # type:ignore
        for asn in all_asns
        if get_ripe_as_country(asn) in di
    ]
    
    plt.figure('feature_democracy_index_dist', figsize=(3.5, 3))
    plt.ecdf(data)
    plt.ylabel('CDF')
    plt.xlabel('Democracy Index')
    plt.tight_layout()
    plt.show()


def _plot_born_date():
    all_asns = set(get_rir_asstats_all().keys())
    rirstats = get_rir_asstats_all()

    data = [
        rirstats[asn]['born'].year for asn in all_asns
        if rirstats[asn]['born'] is not None
    ]
    data = { year: data.count(year) for year in range(1980, 2024) }
    
    plt.figure('feature_rirstat_born_dist', figsize=(5, 4))
    plt.bar(list(data.keys()), list(data.values()))
    plt.ylabel('ASNs Allocated')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.show()


def _research_feature_distribution():
    # _plot_customer_cone()
    # _plot_category_distribution()
    # _plot_democracy_index()
    _plot_born_date()



if __name__ == '__main__': _research_feature_distribution()