from collections import OrderedDict, defaultdict
from datetime import date, datetime
from itertools import pairwise
from pprint import pprint
from statistics import mean
import typing as t
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.service.bgp_markov_chain import get_as_path_confidence, get_link_confidence, get_link_confidence_from_link_counts, markov_chain_from_ripe_ris
from bgpsyche.service.bgp_path_snippet_length import longest_real_snippet_len
from bgpsyche.service.ext import peeringdb, ripe_ris
from bgpsyche.service.ext.asdb import NAICSLITE_SELECTION, get_asdb_full, get_asdb_primary
from bgpsyche.service.ext.asrank import get_asrank_customer_cone_sizes_full
from bgpsyche.service.ext.ripe_as_names_countries import get_ripe_as_country, get_ripe_as_name
from bgpsyche.service.ext.rir_delegations import get_all_asns, get_rir_asstats_all
from bgpsyche.stage2_enrich.as_category import AS_CATEGORY, get_as_category
from bgpsyche.stage2_enrich.democracy_index import get_democracy_index
from bgpsyche.stage2_enrich.global_trade import get_normalized_trade_relationship
from bgpsyche.stage3_rank.make_dataset import make_dataset
from bgpsyche.stage3_rank.vectorize_features import AS_FEATURE_VECTOR_NAMES
from bgpsyche.stage3_rank.vectorize_util import scale_zero_to_one_linear
from bgpsyche.util.cancel_iter import cancel_iter
from bgpsyche.util.geo import ALPHA2_OFFICIAL, COUNTRY_DISTANCES
from .as_graph import as_graph_ris

logging_setup()
_LOG = logging.getLogger(__name__)

@PickleFileCache.decorate
def _ris_paths():
    return [
        meta['path'] for meta in ripe_ris.iter_paths(
            datetime.fromisoformat('2023-05-01T00:00'),
            eliminate_path_prepending=True
        )
    ]


_all_links = PickleFileCache(
    'research_ris_links',
    lambda: set(pairwise(
        asn
        for path in _ris_paths()
        for asn in path
    ))
)

def _plot_customer_cone():
    data = [
        math.log(cone or 1, 10)
        for cone in get_asrank_customer_cone_sizes_full().values()
    ]
    
    plt.figure('feature_cone_cdf', figsize=(3.5, 3))
    plt.ecdf(data)
    plt.ylabel('CDF')
    plt.xlabel('Customer Cone Size ($log_{10}$)')
    plt.xlim([-0.2, 3])
    plt.tight_layout()
    plt.show()


def _plot_longest_real_snippet():
    dataset = make_dataset()

    data = [
        longest_real_snippet_len(el['path'])
        for el in cancel_iter(tqdm(dataset))
    ]
    plt.figure('feature_real_snippet_length_dist', figsize=(3.5, 3))
    plt.xlabel('Real Snippet Length')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ylabel('Amount')
    plt.xlim([-1, 9])
    # plt.hist(data, bins=np.arange(10)-0.5) # type: ignore
    _range = list(range(0,9))
    plt.xticks(_range)
    plt.bar(_range, [ data.count(i) for i in _range ])
    plt.tight_layout()
    plt.show()


def _plot_link_trade_volume():
    links = _all_links.get()
    print(len(links))

    countries = [
        (get_ripe_as_country(src), get_ripe_as_country(dst))
        for src, dst in links
    ]
    print(countries[:10])

    data = [
        get_normalized_trade_relationship(src, dst)
        for src, dst in countries
        if src in ALPHA2_OFFICIAL and dst in ALPHA2_OFFICIAL
    ]

    print(len(data))
    
    plt.figure('feature_trade_factor_dist', figsize=(3.5, 3))
    plt.ylabel('CDF')
    plt.xlabel('Trade Volume,\n Normalized by Total Imports/Exports')
    plt.ecdf(data)
    plt.tight_layout()
    plt.show()

def _plot_path_confidence():
    dt = date.fromisoformat('2023-05-01')

    data = [
        get_as_path_confidence(path, *markov_chain_from_ripe_ris(dt))
        for path in tqdm(_ris_paths())
    ]

    plt.figure('feature_path_confidence_dist', figsize=(3.5, 3))
    plt.ecdf(data)
    plt.xlabel('Confidence')
    plt.ylabel('CDF')
    plt.tight_layout()
    plt.show()


def _plot_link_confidence():
    g = as_graph_ris.get()
    links = set(g.edges)
    print(len(links))


    dt = date.fromisoformat('2023-05-01')

    count_full, count_per_dest = markov_chain_from_ripe_ris(dt)

    data = [
        get_link_confidence(source, sink, count_full, count_per_dest)
        # (count_full[source][sink] / sum(count_full[source].values()))
        for source, sink in links
        # if len(g[source]) > 10
    ]

    print(len(data))

    links_per_as = [ len(g[asn]) for asn in g.nodes ]
    tail = [ lpa for lpa in links_per_as if lpa > 10**2 ]

    _LOG.info(f'tail length: {round(len(tail) / len(links_per_as), 5)}')
    _LOG.info(f'tail influence: {round(sum(tail) / sum(links_per_as), 5)}')

    plt.figure('feature_link_confidence_links_per_as_full', figsize=(4.5, 4))
    plt.hist([ math.log(lpa, 10) for lpa in links_per_as ])
    plt.xlabel('Links in Graph ($log_{10}$)')
    plt.ylabel('Amount of ASes')
    plt.tight_layout()

    plt.figure('feature_link_confidence_links_per_as_tail', figsize=(4.5, 4))
    plt.hist([ math.log(lpa, 10) for lpa in tail ])
    plt.xlabel('Links in Graph ($log_{10}$)')
    plt.ylabel('Amount of ASes')
    plt.tight_layout()

    plt.figure('feature_link_confidence_dist', figsize=(4.5, 4))
    plt.xlabel('Link Confidence')
    plt.ylabel('Amount of Links')
    plt.hist(data, bins=31)
    plt.tight_layout()

    plt.show()


def _plot_category_distribution():

    # pdb_asns = peeringdb.Client.get_all_asns()
    # assigned_asns = set(get_rir_asstats_all().keys())
    # all_asns = assigned_asns.union(pdb_asns)
    all_asns = set(get_rir_asstats_all().keys())

    # all_asns = { asn for el in tqdm(make_dataset()) for asn in el['path'] }

    def mkplot(name: str):
        nonlocal get_cat, labels, labels_rename, figsize
        cats = [ get_cat(asn) for asn in cancel_iter(tqdm(all_asns, name)) ]
        cats = [ cat for cat in cats if cat is not None ]
        _LOG.info(
            f'len cats {name}: ' +
            str(len(cats) - (
                cats.count("Unknown") + cats.count("Other") + cats.count("Not Disclosed")
                + cats.count("Computer and IT - Other")
                + cats.count("Computer and IT - Unknown")
            ))
        )
        data = { cat: (cats.count(cat) / len(cats)) * 100 for cat in labels }
        if name == 'as_category_distribution_asdb':
            _LOG.info(f'ASdb Unknown: {data["Unknown"]}')
            _LOG.info(f'ASdb Other: {data["Other"]}')
            # _LOG.info(f'Assigned len: {len(assigned_asns)}')
            _LOG.info(f'ASdb len: {len(set(get_asdb_full().keys()))}')
            # _LOG.info(
            #     f'ASdb asssigned diff: ' +
            #     f'{len(set(get_asdb_full().keys()).symmetric_difference(assigned_asns))}'
            # )

        labels = sorted(data.keys(), key=lambda cat: data[cat])
        labels = [
            labels_rename[l] if l in labels_rename else l
            for l in labels
        ]

        y = sorted(data.values(), reverse=False)

        plt.figure(name, figsize=figsize)
        plt.barh(labels, y)
        # plt.setp(plt.gca().get_xticklabels(), rotation=90, ha='right')
        plt.xlabel('%')
        plt.tight_layout()

    # --- peeringdb --- 
    def get_cat_pdb(asn):
        pdb = peeringdb.Client.get_network_by_asn(asn)
        return pdb.info_type if pdb else None
    get_cat = get_cat_pdb
    labels = list(t.get_args(peeringdb.NetworkType))
    labels_rename = {}
    figsize = (5, 4)
    mkplot('as_category_distribution_pdb')

    # --- asdb: bgpsyche selection --- 
    get_cat = get_asdb_primary
    labels = list(NAICSLITE_SELECTION)
    labels_rename = {
        'Agriculture, Mining, and Refineries (Farming, Greenhouses, Mining, Forestry, and Animal Farming)': 'Agriculture, Mining, and Refineries',
        'Computer and IT - Internet Service Provider (ISP)': 'Computer and IT - ISP',
        'Computer and IT - Internet Exchange Point (IXP)': 'Computer and IT - IXP',
        'Computer and IT - Hosting, Cloud Provider, Data Center, Server Colocation': 'Computer and IT - Hosting, Cloud Provider, [...]',
    }
    figsize = (7, 6)
    mkplot('as_category_distribution_asdb')

    
    # --- bgpsyche ---
    get_cat = get_as_category
    labels = list(AS_CATEGORY)
    labels_rename = {}
    figsize = (5, 4)
    mkplot('as_category_distribution_bgpsyche')

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


def _plot_distance_from_source_km():

    paths = _ris_paths()
    distances: t.Dict[int, t.List[float]] = defaultdict(list)

    for path in tqdm(paths):
        src = path[0]
        cc_src = get_ripe_as_country(src)
        if cc_src not in ALPHA2_OFFICIAL: continue

        for i in range(1, len(path)):
            cc_curr = get_ripe_as_country(path[i])
            if cc_curr in ALPHA2_OFFICIAL:
                to_dst = len(path) - i
                distances[to_dst].append(COUNTRY_DISTANCES[cc_src][cc_curr])

                
    data = {
        i: mean(d) for i, d in
        tqdm(sorted(distances.items()))
    }
    pprint(data)

    plt.figure('feature_distance_from_path_beginning_km_to_dest', figsize=(3.5, 3))
    pprint((list(data.keys()), list(data.values()),))
    plt.plot(
        list(data.keys()), list(data.values()),
    )
    plt.xticks(list(range(1, 14)))
    plt.ylabel('Mean Distance to Source AS in km')
    plt.xlabel('AS Path Hops to Destination AS')
    plt.tight_layout()
    plt.show()


def _plot_path_length():
    paths = _ris_paths()
    data: t.Dict[int, int] = defaultdict(int)
    for path in tqdm(paths): data[len(path)] += 1

    data = { k: v for k, v in sorted(data.items()) }

    plt.figure('bgp_path_length_distribution', figsize=(4.5, 3))
    plt.bar(list(data.keys()), list(data.values()))
    plt.xlabel('Path Length')
    plt.xticks(list(range(1, 16)))
    plt.ylabel('Amount of Paths')
    plt.tight_layout()
    plt.show()
        


def _research_feature_distribution():
    # _plot_customer_cone()
    # _plot_category_distribution()
    # _plot_democracy_index()
    # _plot_born_date()
    # _plot_longest_real_snippet()
    # _plot_path_length()
    # _plot_link_trade_volume()
    # _plot_link_confidence()
    _plot_path_confidence()
    # _plot_distance_from_source_km()



if __name__ == '__main__': _research_feature_distribution()