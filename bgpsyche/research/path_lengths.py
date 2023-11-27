from collections import defaultdict
from datetime import datetime
from pprint import pformat
import typing as t

import bgpsyche.logging_config
from bgpsyche.service.ext import ripe_ris

def _research_compute_path_lengths():
    res: t.Dict[int, int] = defaultdict(int)
    for path_meta in ripe_ris.iter_paths(
            datetime.fromisoformat('2023-05-01T00:00')
    ):
        l = len(path_meta['path'])
        res[l] += 1

    print(pformat(res))

    total = sum(res.values())

    over_5_i = [ k for k in res.keys() if k > 5]
    over_5 = sum([ res[i] for i in over_5_i ])
    print(over_5 / total)


if __name__ == '__main__': _research_compute_path_lengths()