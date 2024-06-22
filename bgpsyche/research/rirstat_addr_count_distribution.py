from collections import defaultdict
from pprint import pformat
from statistics import mean
import typing as t
from datetime import datetime
import logging

from matplotlib import pyplot as plt
from bgpsyche.service.ext import peeringdb
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.service.ext.rir_delegations import get_rir_asstats_all
from bgpsyche.service.ext import ripe_ris
from bgpsyche.util.run_in_pypy import run_in_pypy

logging_setup()
_LOG = logging.getLogger(__name__)

_DT = datetime.fromisoformat('2023-05-01T00:00')


@run_in_pypy(cache=PickleFileCache)
def _rirstat_addr_count_distribution() -> t.Tuple[
        t.Dict[int, float], t.Dict[int, float]
]:
    rirstats = get_rir_asstats_all(_DT)

    counts_v4: t.Dict[int, t.List[float]] = defaultdict(list)
    counts_v6: t.Dict[int, t.List[float]] = defaultdict(list)


    for path_meta in ripe_ris.iter_paths(_DT, eliminate_path_prepending=True):
        path = path_meta['path']
        path_len = len(path)
        for pos, asn in enumerate(path):
            pos_from_back = path_len - pos
            if asn in rirstats:
                counts_v4[pos_from_back].append(rirstats[asn]['addr_count_v4_log_2'])
                counts_v6[pos_from_back].append(rirstats[asn]['addr_count_v6_log_2'])

    counts_v4_means = { pos: mean(counts) for pos, counts in counts_v4.items() }
    counts_v6_means = { pos: mean(counts) for pos, counts in counts_v6.items() }

    return counts_v4_means, counts_v6_means

def _plot_rirstat_addr_count_distribution() -> None:

    counts_v4_means, counts_v6_means = _rirstat_addr_count_distribution()

    counts_v4_means = dict(sorted(counts_v4_means.items()))
    counts_v6_means = dict(sorted(counts_v6_means.items()))

    plt.figure('feature_rirstat_addr_count_dist_from_dest', figsize=(4, 3))
    plt.xlabel('Distance from Destination AS')
    plt.ylabel('$log_2(\\text{Address Count})$')

    # print(pformat(counts_v4_means))
    plt.plot(
        list(counts_v4_means.keys()), list(counts_v4_means.values()),
        label='IPv4',
    )
    plt.xticks([i for i in range(16)])
    plt.plot(
        list(counts_v6_means.keys()), list(counts_v6_means.values()),
        label='IPv6',
    )
    plt.legend()

    plt.tight_layout()

def _plot_rirstat_addr_count_distribution_full():
    rirstats = get_rir_asstats_all()
    all_asns = set(get_rir_asstats_all().keys())

    plt.figure('feature_rirstat_addr_count_dist', figsize=(4, 3))
    plt.xlabel('$log_2(\\text{Address Count})$')
    plt.ylabel('CDF')

    data_v4 = [ rirstats[asn]['addr_count_v4_log_2'] for asn in all_asns ]
    data_v6 = [ rirstats[asn]['addr_count_v6_log_2'] for asn in all_asns ]

    print(data_v4.count(0), data_v4.count(1), data_v4.count(2))

    plt.ecdf(data_v4, label='IPv4')
    plt.ecdf(data_v6, label='IPv6')
    plt.legend(loc='lower right')

    plt.tight_layout()


if __name__ == '__main__':
    _plot_rirstat_addr_count_distribution()
    _plot_rirstat_addr_count_distribution_full()
    plt.show()