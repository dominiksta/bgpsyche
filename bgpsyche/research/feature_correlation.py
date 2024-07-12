from collections import defaultdict
from functools import reduce
from itertools import combinations, groupby
import math
from pprint import pprint
from statistics import correlation
import typing as t

import numpy as np
import scipy.stats as ss
from matplotlib import pyplot as plt
from tqdm import tqdm
from dython.nominal import correlation_ratio
from bgpsyche.research.heatmaps import annotate_heatmap, heatmap
from bgpsyche.service.ext.asrank import get_asrank, get_asrank_customer_cone_size, get_asrank_customer_cone_sizes_full, get_asrank_customer_cones
from bgpsyche.service.ext.rir_delegations import get_rir_asstats, get_rir_asstats_all
from bgpsyche.stage3_rank.make_dataset import DatasetEl, make_dataset
from bgpsyche.stage3_rank.vectorize_features import AS_FEATURE_VECTOR_NAMES, LINK_FEATURE_VECTOR_NAMES, PATH_FEATURE_VECTOR_NAMES

"""
A note on the statistics:
When computing correlations between variables, one must honour their 'type',
that is if they are categorical or numerical.

- numerical   - numerical     : pearsons r (statistics.correlation)
- numerical   - binary        : pearsons r (statistics.correlation)
- binary      - binary        : pearsons r (statistics.correlation)
- categorical - categorical   : cramers v  (scipy.stats.contingency.association)
- categorical - numerical     : 'eta'      (dython.nominal.correlation_ratio)

Also, do be careful when thinking about one-hot encoding a category:
- The new individual binary variables can be compared to other variables
  according to the list above.
- But: It makes no sense to compare the individual binary variables with each
  other. There is by definition no correlation because they can never be '1' at
  the same time, but computing correlation using pearsons r or similar will
  still yield nonsense non-zero values. Just set these to zero.

There are other algorithms/formulas, but these seem to be the most common. I did
not major in statistics :p
"""

_FeatureType = t.Literal['categorical', 'numerical', 'binary']
_FeaturesDesc = t.Dict[str, _FeatureType]
_FeaturesIn = t.Dict[str, t.List]

def _all_equal(iterable: t.Iterable) -> bool:
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def _compute_correlations(
        desc: _FeaturesDesc,
        features: _FeaturesIn,
) -> t.Dict[str, t.Dict[str, float]]:
    ret: t.Dict[str, t.Dict[str, float]] = defaultdict(dict)

    for f in features.keys(): ret[f][f] = 1

    def _binary_vars_in_same_cat(f1: str, f2: str) -> bool:
        d1, d2 = desc[f1], desc[f2]
        if d1 != 'binary' or d2 != 'binary': return False
        def get_cat(f: str) -> str:
            split = f1.split('=')
            assert len(split) == 2, \
                'binary categories must be of the shape category_name=category_instance'
            return split[0]
        return get_cat(f1) == get_cat(f2)

    for f1, f2 in tqdm(combinations(features.keys(), 2), 'compute correlations'):
        # correlation is not defined when one feature is constant, so we set it to 0
        if _all_equal(features[f1]) or _all_equal(features[f2]):
            ret[f1][f2] = 0
        else:
            assert f1 in desc and f2 in desc

            if _binary_vars_in_same_cat(f1, f2):
                ret[f1][f2] = 0 # not defined
            elif desc[f1] == 'categorical' and desc[f2] == 'categorical':
                crosstab = ss.contingency.crosstab(features[f1], features[f2])
                ret[f1][f2] = ss.contingency.association(crosstab, method='cramer')
            elif desc[f1] == 'categorical' and desc[f2] == 'numerical':
                ret[f1][f2] = correlation_ratio(features[f1], features[f2])
            else:
                ret[f1][f2] = round(correlation(features[f1], features[f2]), 2)

        ret[f2][f1] = -ret[f1][f2]

    return dict(ret)

def _plot_correlations(
        corrs: t.Dict[str, t.Dict[str, float]],
        name: str, figsize: t.Tuple[float, float],
):
    pprint(corrs)
    corrs_mat = np.array([
        [ abs(corrs[f1][f2]) for f2 in corrs.keys() ]
        for f1 in corrs.keys()
    ])
    plt.figure(name, figsize)

    im, _ = heatmap(
        corrs_mat, list(corrs.keys()), list(corrs.keys()),
        ax=plt.gca(), cmap="magma_r", cbarlabel="Correlation"
    )
    annotate_heatmap(
        im, textcolors=lambda val: 'black' if val < 0.25 else 'white',
        valfmt=lambda x: '1.0' if x == 1 else f'{x:.2f}'[1:]
    )

    plt.tight_layout()
    plt.show()

    # plt.imshow(corrs_mat, cmap='hot')

    # plt.xticks(np.arange(len(corrs)), labels=list(corrs.keys()))
    # plt.yticks(np.arange(len(corrs)), labels=list(corrs.keys()))
    # plt.setp(
    #     plt.gca().get_xticklabels(), rotation=45, ha='right',
    #     rotation_mode="anchor"
    # )

    # for i in range(len(corrs)):
    #     for j in range(len(corrs)):
    #         plt.gca().text(
    #             j, i, str(corrs_mat[i][j]), ha='center', va='center',
    #             color='black' if corrs_mat[i][j] > 0.75 else 'white'
    #         )

    # plt.tight_layout()
    # plt.show()

def _plot_as_feature_correlation(dataset: t.List[DatasetEl]):

    as_feature_types: _FeaturesDesc = {
        'Customer Cone'            : 'numerical',
        'ASN Registered Date'      : 'numerical',
        'v4 Addresses'             : 'numerical',
        'v6 Addresses'             : 'numerical',
        'Dist. from Path Source'   : 'numerical',
        'Cat=Unknown'              : 'binary',
        'Cat=Transit/Access'       : 'binary',
        'Cat=Content'              : 'binary',
        'Cat=Enterprise'           : 'binary',
        'Cat=Educational/Research' : 'binary',
        'Cat=Non-Profit'           : 'binary',
        'Cat=Route Server'         : 'binary',
        'Cat=Network Services'     : 'binary',
        'Cat=Route Collector'      : 'binary',
        'Cat=Government'           : 'binary',
        'Democracy Index'          : 'numerical',
    }

    name_map = {
        'as_rank_cone'                    : 'Customer Cone',
        'rirstat_born'                    : 'ASN Registered Date',
        'rirstat_addr_count_v4'           : 'v4 Addresses',
        'rirstat_addr_count_v6'           : 'v6 Addresses',
        'distance_from_path_beginning_km' : 'Dist. from Path Source',
        'category_unknown'                : 'Cat=Unknown',
        'category_transit_access'         : 'Cat=Transit/Access',
        'category_content'                : 'Cat=Content',
        'category_enterprise'             : 'Cat=Enterprise',
        'category_educational_research'   : 'Cat=Educational/Research',
        'category_non_profit'             : 'Cat=Non-Profit',
        'category_route_server'           : 'Cat=Route Server',
        'category_network_services'       : 'Cat=Network Services',
        'category_route_collector'        : 'Cat=Route Collector',
        'category_government'             : 'Cat=Government',
        'country_democracy_index'         : 'Democracy Index',
    }

    feat_per_as = {}
    for el in tqdm(dataset, 'feat per as'):

        for asf in el['as_features']:
            assert asf[5:15].count(1) == 1 and asf[5:15].count(0) == 9, (
                'category is not binary !?', el
            )

        for i  in range(len(el['path'])):
            if el['path'][i] in feat_per_as: continue
            feat_per_as[el['path'][i]] = el['as_features'][i]

    as_feature_names = [ name_map[f] for f in AS_FEATURE_VECTOR_NAMES ]
    as_features: _FeaturesIn = {
        as_feature_names[i]: [ per_as[i] for per_as in feat_per_as.values() ]
        for i in tqdm(range(len(as_feature_names)), 'dictify')
    }

    _plot_correlations(
        _compute_correlations(as_feature_types, as_features),
        'feature_correlation_as_full', figsize=(8, 6)
    )


def _plot_link_feature_correlation(dataset: t.List[DatasetEl]):

    name_map = {
        'rel_p2c'                    : 'Relation=P2C',
        'rel_p2p'                    : 'Relation=P2P',
        'rel_c2p'                    : 'Relation=C2P',
        'rel_unknown'                : 'Relation=Unknown',
        'distance_km'                : 'Geographic Distance',
        'trade_factor'               : 'Trade Volume',
        'confidence_from_seen_count' : 'Confidence by Seen Count',
    }

    link_feature_types: _FeaturesDesc = {
        'Relation=P2C'             : 'binary',
        'Relation=P2P'             : 'binary',
        'Relation=C2P'             : 'binary',
        'Relation=Unknown'         : 'binary',
        'Geographic Distance'      : 'numerical',
        'Trade Volume'             : 'numerical',
        'Confidence by Seen Count' : 'numerical',
    }

    feat_per_link: t.Dict[str, t.List] = {}
    for el in tqdm(dataset, 'feat per link'):
        for i in range(len(el['path']) - 1):
            source, sink = el['path'][i], el['path'][i+1]
            key = f'{source}->{sink}'
            if key in feat_per_link: continue
            feat_per_link[key] = el['link_features'][i]

    link_feature_names = [ name_map[f] for f in LINK_FEATURE_VECTOR_NAMES ]
    link_features: _FeaturesIn = {
        link_feature_names[i]: [ per_link[i] for per_link in feat_per_link.values() ]
        for i in tqdm(range(len(link_feature_names)), 'dictify')
    }

    _plot_correlations(
        _compute_correlations(link_feature_types, link_features),
        'feature_correlation_link_full', figsize=(6.5, 5)
    )


def _plot_path_feature_correlation(dataset: t.List[DatasetEl]):

    name_map = {
        'length'                     : 'Length',
        'is_valley_free'             : 'Valley-Free',
        'longest_real_snippet'       : 'Longest Real Snippet',
        'per_dest_markov_confidence' : 'Confidence by Link Seen Count',
        'is_real'                    : 'Is Real/Correct',
    }

    path_feature_types: _FeaturesDesc = {
        'Length'                        : 'numerical',
        'Valley-Free'                   : 'numerical',
        'Longest Real Snippet'          : 'numerical',
        'Confidence by Link Seen Count' : 'numerical',
        'Is Real/Correct'               : 'numerical',
    }

    feat_per_path: t.Dict[str, t.List] = {}
    for el in tqdm(dataset, 'feat per path'):
        key = '->'.join(map(str, el['path']))
        if key in feat_per_path: continue
        feat_per_path[key] = [ *el['path_features'], int(el['real']) ]

    path_feature_names = \
        [ name_map[f] for f in [ *PATH_FEATURE_VECTOR_NAMES, 'is_real' ] ]
    path_features: _FeaturesIn = {
        path_feature_names[i]: [ per_path[i] for per_path in feat_per_path.values() ]
        for i in tqdm(range(len(path_feature_names)), 'dictify')
    }

    _plot_correlations(
        _compute_correlations(path_feature_types, path_features),
        'feature_correlation_path', figsize=(6.5, 4)
    )


def _research_feature_correlation():
    dataset = make_dataset()
    # _plot_as_feature_correlation(dataset)
    # _plot_link_feature_correlation(dataset)
    _plot_path_feature_correlation(dataset)

if __name__ == '__main__': _research_feature_correlation()
