from collections import defaultdict
from pprint import pformat
from statistics import mean
import typing as t
from datetime import datetime
import logging

from matplotlib import pyplot as plt
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.service.ext.rir_delegations import get_rir_asstats_all
from bgpsyche.service.ext import routeviews

logging_setup()
_LOG = logging.getLogger(__name__)

def _rirstat_addr_count_distribution(dt: datetime) -> t.Tuple[
        t.Dict[int, float], t.Dict[int, float]
]:
    rirstats = get_rir_asstats_all(dt)

    counts_v4: t.Dict[int, t.List[float]] = defaultdict(list)
    counts_v6: t.Dict[int, t.List[float]] = defaultdict(list)


    for path_meta in routeviews.iter_paths(dt, eliminate_path_prepending=True):
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

def _plot_rirstat_addr_count_distribution(dt: datetime) -> None:

    cache = PickleFileCache(
        'research_rirstat_addr_count_distributions',
        lambda: _rirstat_addr_count_distribution(dt),
    )
    # cache.invalidate()
    counts_v4_means, counts_v6_means = cache.get()

    counts_v4_means = dict(sorted(counts_v4_means.items()))
    counts_v6_means = dict(sorted(counts_v6_means.items()))

    fig = plt.figure(figsize=(9, 4), layout="constrained")
    axs = t.cast(t.Any, fig.subplots(1, 2))
    fig.supxlabel('Distance from Destination AS')
    fig.supylabel('$log_2(\\text{adress_count})$')

    # print(pformat(counts_v4_means))
    axs[0].plot(list(counts_v4_means.keys()), list(counts_v4_means.values()))
    axs[0].set_ylim([0, 35])
    axs[0].set_title('IPv4')
    axs[1].plot(list(counts_v6_means.keys()), list(counts_v6_means.values()))
    axs[1].set_ylim([0, 35])
    axs[1].set_title('IPv6')

    plt.show()

if __name__ == '__main__':
    dt = datetime.fromisoformat('2023-05-01T00:00')

    _plot_rirstat_addr_count_distribution(dt)