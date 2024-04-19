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
from bgpsyche.stage3_rank.make_dataset import make_dataset
from bgpsyche.stage3_rank.vectorize_features import AS_FEATURE_VECTOR_NAMES, PATH_FEATURE_VECTOR_NAMES

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

def _plot_correlations(corrs: t.Dict[str, t.Dict[str, float]]):
    pprint(corrs)
    corrs_mat = np.array([
        [ abs(corrs[f1][f2]) for f2 in corrs.keys() ]
        for f1 in corrs.keys()
    ])
    fig, ax = plt.subplots()

    im, _ = heatmap(
        corrs_mat, list(corrs.keys()), list(corrs.keys()),
        ax=ax, cmap="magma_r", cbarlabel="correlation"
    )
    annotate_heatmap(
        im, textcolors=lambda val: 'black' if val < 0.25 else 'white',
        valfmt=lambda x: '1.0' if x == 1 else f'{x:.2f}'[1:]
    )

    fig.tight_layout()
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


def _research_feature_correlation():

    dataset = make_dataset()


    # path_features: _FeaturesIn = {
    #     PATH_FEATURE_VECTOR_NAMES[i]: [ el['path_features'][i] for el in dataset ]
    #     for i in tqdm(range(len(PATH_FEATURE_VECTOR_NAMES)), 'dictify')
    # }
    # path_features['is_real'] = [ int(el['real']) for el in dataset ]

    # _plot_correlations(_compute_correlations(path_features))

    as_feature_types: _FeaturesDesc = {
        'as_rank_cone'             : 'numerical',
        'rirstat_born'             : 'numerical',
        'rirstat_addr_count_v4'    : 'numerical',
        'rirstat_addr_count_v6'    : 'numerical',
        'cat=unknown'              : 'binary',
        'cat=transit_access'       : 'binary',
        'cat=content'              : 'binary',
        'cat=enterprise'           : 'binary',
        'cat=educational_research' : 'binary',
        'cat=non_profit'           : 'binary',
        'cat=route_server'         : 'binary',
        'cat=network_services'     : 'binary',
        'cat=route_collector'      : 'binary',
        'cat=government'           : 'binary',
        'country_democracy_index'  : 'numerical',
    }

    feat_per_as = {}
    for el in tqdm(dataset, 'feat per as'):

        for asf in el['as_features']:
            assert asf[4:14].count(1) == 1 and asf[4:14].count(0) == 9, el

        for i  in range(len(el['path'])):
            if el['path'][i] in feat_per_as: continue
            feat_per_as[el['path'][i]] = el['as_features'][i]

    as_feature_names = \
        [ f.replace('category_', 'cat=') for f in AS_FEATURE_VECTOR_NAMES ]
    as_features: _FeaturesIn = {
        as_feature_names[i]: [ per_as[i] for per_as in feat_per_as.values() ]
        for i in tqdm(range(len(as_feature_names)), 'dictify')
    }

    # all_asns = list(get_asrank_customer_cone_sizes_full().keys())
    # rirstats = get_rir_asstats_all()

    # as_features: _FeaturesIn = {
    #     'asrank_cone': [ math.log(get_asrank_customer_cone_size(asn) or 1) for asn in all_asns ],
    #     'asrank': [ get_asrank(asn) for asn in all_asns ],
    #     'rirstat_v4': [ rirstats[asn]['addr_count_v4_log_2'] if asn in rirstats else 0 for asn in all_asns ],
    #     'rirstat_v6': [ rirstats[asn]['addr_count_v6_log_2'] if asn in rirstats else 0 for asn in all_asns ],
    # }

    _plot_correlations(_compute_correlations(as_feature_types, as_features))


if __name__ == '__main__': _research_feature_correlation()
