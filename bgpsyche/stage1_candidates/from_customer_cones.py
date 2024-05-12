from collections import defaultdict
from datetime import datetime
from functools import reduce
import itertools
import logging
import typing as t

import networkx as nx
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.service.bgp_graph import as_graph_from_ext
from bgpsyche.service.ext.asrank import get_asrank_customer_cones_full
from bgpsyche.service.ext.caida_asrel import get_caida_asrel
from bgpsyche.stage1_candidates.from_graph import (
    GetPathCandidatesAbortCondition, PathCandidatesRes, abort_on_amount,
    abort_on_timeout, get_path_candidates_from_graph
)
from bgpsyche.util.benchmark import Progress, bench_end, bench_start
from bgpsyche.util.bgp.relationship import Source2Sink2Rel
from bgpsyche.util.bgp.tier1 import TIER1_SIBLINGS_FLAT
from bgpsyche.util.run_in_pypy import run_in_pypy

logging_setup()
_LOG = logging.getLogger(__name__)

Upstreams = t.Dict[int, t.Set[int]]

def _compute_approx_upstreams_from_relationships(
        rels: Source2Sink2Rel
) -> Upstreams:
    ret: Upstreams = defaultdict(set)

    # All t1 ASes are set to have each other as "upstreams" here. While not
    # technically correct, this is necessary for the path search algo in this
    # file.
    for asn in TIER1_SIBLINGS_FLAT:
        ret[asn] = TIER1_SIBLINGS_FLAT.difference({ asn })

    todo = set(rels.keys()).difference(TIER1_SIBLINGS_FLAT)

    prg_interval = 100
    prg_iter = 0
    prg = Progress(int(len(todo) / prg_interval), 'approx_upstreams_for_rels')

    while len(todo) > 0:
        prg_iter += 1
        if prg_iter % prg_interval == 0: prg.update()

        curr_asn = -1
        direct_c2p: t.Set[int] = set()

        # if prg_iter % prg_interval == 0: bench_find_next = bench_start('find_next')
        for asn in todo:
            direct_c2p = {
                peer for peer in rels[asn] if rels[asn][peer] == 'c2p'
            }
            if len(direct_c2p.intersection(todo)) != 0: continue
            curr_asn = asn
            break
        # if prg_iter % prg_interval == 0: bench_end(bench_find_next)
        

        assert curr_asn != -1

        ret[curr_asn] = reduce(
            set.union,
            itertools.chain(
                [ direct_c2p ],
                ( ret[upstream] for upstream in direct_c2p ),
            ),
            set()
        )
        # if prg_iter % prg_interval == 0:
        #     _LOG.info(f'upstreams of {curr_asn}: {ret[curr_asn]}')
        #     _LOG.info(f'upstreams of {curr_asn}: {direct_c2p}')
        todo.remove(curr_asn)

    prg.complete()

    for asn in rels.keys(): ret[asn] = ret[asn].union(TIER1_SIBLINGS_FLAT)

    return dict(ret)


def _bgp_subgraph_for_path_candidates_from_customer_cones(
        source: int, sink: int,
        full_graph: nx.Graph,
        # rels: Source2Sink2Rel,
        customer_cones: t.Dict[int, t.Set[int]],
) -> nx.Graph:
    g = nx.Graph()

    subgraph_nodes: t.List[int] = [source, sink] + [
        asn for asn in full_graph.nodes
        if asn in customer_cones and (
                source in customer_cones[asn] or
                sink in customer_cones[asn]
        )
    ]

    for asn in subgraph_nodes: g.add_node(asn)

    for asn in subgraph_nodes:
        for peer in full_graph[asn]:
            if peer not in g: continue
            g.add_edge(asn, peer)

    return g



@run_in_pypy(cache=PickleFileCache)
def _test():
    dt = datetime.fromisoformat('2023-05-01T00:00')
    upstreams = _compute_approx_upstreams_from_relationships(get_caida_asrel(dt))
    return upstreams


def get_path_candidates_from_customer_cones(
        source: int, sink: int,
        abort_on: t.Callable[
            [], t.List[GetPathCandidatesAbortCondition]
        ] = lambda: [
            { 'func': abort_on_timeout(1), 'desc': 'timeout 1s' },
            { 'func': abort_on_amount(1000), 'desc': 'amount 1k' },
        ],
        quiet: bool = False,
) -> PathCandidatesRes:
    return get_path_candidates_from_graph(
        _bgp_subgraph_for_path_candidates_from_customer_cones(
            source, sink,
            as_graph_from_ext(),
            get_asrank_customer_cones_full(),
        ),
        source, sink, abort_on, quiet,
    )

    

if __name__ == '__main__':
    upstreams = _test()
    # print(pformat(upstreams[3320]))
    # print(pformat(upstreams[51378])) # klinikum ingo
    # print(pformat(get_asrank_customer_cones(51378)))
    # print(pformat(get_asrank_customer_cones(3320)))

    source2sink_real: t.Dict[int, t.Tuple[int, t.List[int]]] = {
        23673: ( 24371, [ 23673, 23764, 4134, 4538, 23910, 24371 ] ),
        14840: ( 265620, [ 14840, 32098, 13999, 265620 ] ),
    }

    for source, (sink, real) in source2sink_real.items():
        subgraph = _bgp_subgraph_for_path_candidates_from_customer_cones(
            source, sink,
            as_graph_from_ext(),
            get_asrank_customer_cones_full(),
        )
        _LOG.info(f'Subgraph size: {len(subgraph)}')
        for asn in real:
            if asn not in subgraph:
                _LOG.warning(f'real path asn {asn} not in subgraph')

        found = get_path_candidates_from_customer_cones(
            source, sink, abort_on=lambda: [
                { 'func': lambda path: path == real, 'desc': 'path eq' },
                { 'func': abort_on_timeout(1), 'desc': 'timeout 1s' },
                { 'func': abort_on_amount(1000), 'desc': 'amount 1k' },
            ]
        )['candidates']
        print(real in found)

