import json
from glob import glob
from pathlib import Path
from pprint import pprint

import matplotlib
from matplotlib import pyplot as plt
from bgpsyche.research.candidates_include_real import _RESULT_DIR

matplotlib.rcParams.update({'font.size': 14})

if __name__ == '__main__':
    available = sorted(
        glob(f'{_RESULT_DIR}/*candidates-include-real*.json'), reverse=True
    )

    print('options: ')
    pprint([ Path(p).name for p in available ])

    res_file = input('res file: ')
    if res_file != '': res_file = str(_RESULT_DIR / res_file)
    else: res_file = available[0]

    print(f'res_file: {res_file}')

    with open(res_file) as f: data = json.loads(f.read())

    print(f'amount of found: {len(data["took"])}')

    plt.figure(0)
    # plt.title('Time Until Real Path Found')
    plt.ylabel('CDF')
    plt.xlabel('Seconds')
    plt.xlim([0, 10])
    plt.ecdf(data['took'])

    plt.figure(1)
    # plt.title('Position of Real Path')
    plt.ylabel('CDF')
    plt.xlabel('Position')
    plt.xlim([0, 2000])
    plt.ecdf(data['included_pos'])

    plt.figure(2)
    # plt.title('Length of Paths not Found')
    plt.ylabel('Amount')
    plt.xlabel('Length')
    plt.xlim([0, 10])
    plt.hist(data['not_included_len'])

    plt.show()