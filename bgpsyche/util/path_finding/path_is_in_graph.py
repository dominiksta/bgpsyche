import typing as t

from .typ import Graph

_HashableT = t.TypeVar('_HashableT', bound=t.Hashable)

def path_is_in_graph(graph: Graph[_HashableT], path: t.List[_HashableT]) -> bool:
    for i in range(0, len(path) - 1):
        pair = path[i], path[i+1]
        if not pair[0] in graph:
            # print(f'{pair[0]} not in graph')
            return False
        if not pair[1] in graph[pair[0]]:
            # print(f'{pair[1]} not in graph[{pair[0]}]')
            return False
    return True
