from collections import defaultdict
import logging
import typing as t
from time import time

from bgpsyche.util.benchmark import bench_function
from .path_is_in_graph import path_is_in_graph
from .typ import Graph

_LOG = logging.getLogger(__name__)

_HashableT = t.TypeVar('_HashableT', bound=t.Hashable)


@bench_function
def _breadth_first_search(
        graph: Graph[_HashableT], sink: _HashableT,
        predicates_path: t.List[t.Callable[[t.List[_HashableT]], bool]] = [],
        timeout_seconds: float = -1,
) -> t.Dict[_HashableT, t.List[t.List[_HashableT]]]:
    if sink not in graph: return {}

    stack: t.List[_HashableT] = list(graph[sink])
    visited: t.Set[_HashableT] = set()
    out: t.Dict[_HashableT, t.List[t.List[_HashableT]]] = defaultdict(list)

    time_start = time(); iter = 0

    def check_predicates(path: t.List[_HashableT]) -> bool:
        for predicate in predicates_path:
            if not predicate(path): return False
        return True

    while len(stack) > 0:
        current_node = stack.pop()
        if current_node in visited: continue
        visited.add(current_node)

        for neighbour in graph[current_node]:
            if neighbour == sink: out[current_node].append([sink])
            if neighbour in out:
                for path in out[neighbour]:
                    new_path = [neighbour] + path
                    if check_predicates(new_path):
                        pass
                        out[current_node].append(new_path)
            if neighbour not in visited:
                # if neighbour == 32098: print('HIT')
                stack.append(neighbour)

        iter += 1
        if iter % 10 == 0: print(len(stack))
        if iter % 100 == 0: _LOG.info(f'Searched {len(visited)} ASes')
        if timeout_seconds != -1 and \
           time() - time_start > timeout_seconds:
            _LOG.info(
                f'Breadth first search timed out after {timeout_seconds}s'
            )
            break

    for source, paths in out.items():
        out[source] = [[source] + path for path in paths]

    if __debug__:
        _LOG.info('Checking paths are actually in graph')
        for source, paths in out.items():
            for path in paths: assert path_is_in_graph(graph, path), path

    return out


def breadth_first_search(
        graph: Graph[_HashableT], source: _HashableT, sink: _HashableT,
        predicates_path: t.List[t.Callable[[t.List[_HashableT]], bool]] = [],
        timeout_seconds: float = -1
) -> t.List[t.List[_HashableT]]:
    if source not in graph: return []

    return _breadth_first_search(
        graph, sink, predicates_path, timeout_seconds
    )[source]