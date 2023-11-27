import typing as t
import logging

from bgpsyche.util.path_finding.breadth_first import Graph

_LOG = logging.getLogger(__name__)

_T = t.TypeVar('_T')
_HashableT = t.TypeVar('_HashableT', bound=t.Hashable)
_INFINITY = 1_000_000

GraphWithCost = t.Dict[_T, t.Dict[_T, int]]

def dijkstra(
        graph: GraphWithCost[_T], start: _T,
        test_no_order = False
) -> t.Dict[_T, int]:
    nodes = list(graph.keys())
    _LOG.info(f'Dijkstra started on graph with {len(nodes)} nodes')

    distance: t.Dict[_T, int] = { node: _INFINITY for node in nodes }
    final = { node: False for node in nodes }

    distance[start] = 0
    final[start] = True

    # current_node_i = nodes.index(start)
    current_node = start
    iter = 0

    while False in final.values():
        iter += 1
        # current_node = nodes[current_node_i]
        # print(f'current_node: {current_node}')

        for node in graph[current_node]:
            distance[node] = min(
                distance[node],
                distance[current_node] + graph[current_node][node]
            )

        # next_node_i = (current_node_i + 1) % len(nodes)
        # # print(f'next node candidate: {nodes[next_node_i]}')

        # for k in range(0, len(nodes)):
        #     n = nodes[k]
        #     if not final[n] and n != current_node and \
        #        distance[n] < distance[nodes[next_node_i]]:
        #         next_node_i = k

        next_node: t.Optional[_T] = None
        if not test_no_order:
            for node, dist in distance.items():
                if final[node] or node == current_node: continue
                if next_node is None: next_node = node
                if distance[next_node] > dist: next_node = node
        else:
            for node, node_final in final.items():
                if not node_final:
                    next_node = node
                    break

        if iter % 1000 == 0:
            _LOG.info(f'Dijkstra handled nodes: {list(final.values()).count(True)}')

        # print(f'next node actual: {nodes[next_node_i]}')
        # current_node_i = next_node_i
        current_node = t.cast(_T, next_node)
        assert not final[current_node]
        final[current_node] = True


        # print(distance)
        # print(final)

    return distance


def flood_fill_distances(
        graph: Graph[_HashableT], start: _HashableT
) -> t.Dict[_HashableT, int]:

    stack: t.List[_HashableT] = [start]
    visited: t.Set[_HashableT] = set()
    distance: t.Dict[_HashableT, int] = { start: 0 }
    iter = 0

    while len(stack) > 0:
        current_node = stack.pop()
        if current_node in visited: continue
        iter += 1
        visited.add(current_node)
        # print(f'current node: {current_node}')

        for neighbour in graph[current_node]:
            if neighbour == start: distance[current_node] = 1
            if neighbour in distance:
                distance[neighbour] = min(
                    distance[neighbour],
                    distance[current_node] + 1
                )
            else: distance[neighbour] = distance[current_node] + 1

            if neighbour not in visited: stack.append(neighbour)

        if iter % 1000 == 0: _LOG.info(f'Flood fill handled nodes: {iter}')

    # print(distance)
    return distance


if __name__ == '__main__':

    g: GraphWithCost[str] = {
        'a': {          'b': 1,  'c': 13,          'e': 29, 'f': 11, },
        'b': { 'a': 1,           'c': 2,  'd': 17,          'f': 31, },
        'c': { 'a': 13, 'b': 2,           'd': 3,  'e': 21,          },
        'd': {                   'c': 3,           'e': 5,  'f': 23, },
        'e': { 'a': 29,          'c': 21, 'd': 5,           'f': 7   },
        'f': { 'a': 11, 'b': 31,          'd': 23, 'e': 7 }
    }
    g_all_one: GraphWithCost[str] = {
        source: { sink: 1 for sink in sink2count.keys() }
        for source, sink2count in g.items()
    }

    print(dijkstra(g, 'a'))
    print(dijkstra(g_all_one, 'a'))
    print(flood_fill_distances({
        source: set(sink2count.keys()) for source, sink2count in g.items()
    }, 'a'))

    pass