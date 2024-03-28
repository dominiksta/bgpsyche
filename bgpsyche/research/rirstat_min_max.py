from datetime import date, datetime
import logging
from time import mktime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from bgpsyche.logging_config import logging_setup
from bgpsyche.service.ext.rir_delegations import get_rir_asstats_all

logging_setup()
_LOG = logging.getLogger(__name__)

def _research_rirstat_min_max():
    stats = get_rir_asstats_all(dt = date.fromisoformat('2024-03-26'))

    born = [ el['born'] for el in stats.values() if el['born'] is not None ]
    born_min = min(born)
    born_max = max(born)
    _LOG.info(f'born_min: {born_min}')
    _LOG.info(f'born_max: {born_max}')
    plt.figure(1)
    plt.title('born')
    plt.xticks([])
    plt.axes().xaxis.set_major_locator(mdates.YearLocator())
    plt.axes().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=90)
    plt.ecdf(mdates.date2num(born))

    _LOG.info(f'no born stat: {len(stats) - len(born)}')

    v4 = [el['addr_count_v4_log_2'] for el in stats.values()]
    v4_min = min(v4)
    v4_max = max(v4)
    _LOG.info(f'v4_min: {v4_min}')
    _LOG.info(f'v4_max: {v4_max}')
    plt.figure(2)
    plt.title('v4')
    # plt.xticks([0, 1, 5, 10, 15, 25, 30])
    # plt.xlim([0, .25 * 10**8])
    plt.ecdf(v4)


    v6 = [el['addr_count_v6_log_2'] for el in stats.values()]
    v6_min = min(v6)
    v6_max = max(v6)
    _LOG.info(f'v6_min: {v6_min}')
    _LOG.info(f'v6_max: {v6_max}')
    plt.figure(3)
    plt.title('v6')
    plt.ecdf(v6)


    plt.show()


if __name__ == '__main__': _research_rirstat_min_max()