import typing as t
import itertools

from bgpsyche.stage1_candidates.from_customer_cones import (
    get_path_candidates_from_customer_cones
)
from bgpsyche.stage1_candidates.from_full_graph import get_path_candidates_full_graph
from bgpsyche.stage1_candidates.from_graph import (
    GetPathCandidatesAbortConditions, abort_on_amount, abort_on_timeout
)


def get_path_candidates(
        source: int, sink: int,
        abort_customer_cone_search: GetPathCandidatesAbortConditions = lambda: [
            { 'func': abort_on_timeout(1), 'desc': 'timeout 1s [cone]' },
            { 'func': abort_on_amount(1000), 'desc': 'amount 1k [cone]' },
        ],
        abort_full_search: GetPathCandidatesAbortConditions = lambda: [
            { 'func': abort_on_timeout(3), 'desc': 'timeout 3s [full]' },
            { 'func': abort_on_amount(800), 'desc': 'amount 800 [full]' },
        ],
        quiet: bool = False,
) -> t.List[t.List[int]]:
    ret: t.List[t.List[int]] = []
    
    for path in itertools.chain(
            get_path_candidates_from_customer_cones(
                source, sink, abort_on=abort_customer_cone_search, quiet=quiet,
            )['candidates'],
            get_path_candidates_full_graph(
                source, sink, abort_on=abort_full_search, quiet=quiet,
            )['candidates'],
    ):
        if path not in ret: ret.append(path)

    return ret