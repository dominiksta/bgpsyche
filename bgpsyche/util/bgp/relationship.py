from collections import defaultdict
import typing as t

RelationshipKind = t.Literal['c2p', 'p2p', 'p2c']
RELATIONSHIP_KIND: t.Set[RelationshipKind] = set(t.get_args(RelationshipKind))
RelationshipKindOpt = t.Literal[RelationshipKind, '???']
Relationship = t.Tuple[int, int, RelationshipKind]
Source2Sink2Rel = t.Dict[int, t.Dict[int, RelationshipKind]]

def relationship_reverse(rel: RelationshipKind) -> RelationshipKind:
    return t.cast(t.Dict[RelationshipKind, RelationshipKind], {
        'c2p': 'p2c', 'p2c': 'c2p', 'p2p': 'p2p'
    })[rel]


def flatten_source2sink2rel(
        source2sink2rel: Source2Sink2Rel
) -> t.List[Relationship]:
    ret = []
    for source, sink2rel in source2sink2rel.items():
        for sink, rel in sink2rel.items():
            ret.append((source, sink, rel))
    return ret


class MergeSource2Sink2RelRet(t.TypedDict):
    s2s2r: Source2Sink2Rel
    added: int
    changed: int
    confirmed: int

def merge_s2s2r(
        base: Source2Sink2Rel, extend: Source2Sink2Rel,
        disable_add_new = False, disable_overwrite = False,
) -> MergeSource2Sink2RelRet:
    out: MergeSource2Sink2RelRet = {
        's2s2r': defaultdict(dict),
        'added': 0, 'changed': 0, 'confirmed': 0,
    }
    would_add, would_change = 0, 0

    # copy base into out['s2s2r']
    for source, sink2rel in base.items():
        for sink, rel in sink2rel.items():
            out['s2s2r'][source][sink] = rel

    # actual merge
    for source, sink2rel in extend.items():
        for sink, rel in sink2rel.items():
            if source not in base or sink not in base[source]:
                would_add += 1
                if not disable_add_new: out['added'] += 1
                else: continue
            elif base[source][sink] != rel:
                would_change += 1
                if not disable_overwrite: out['changed'] += 1
                else: continue
            else: out['confirmed'] += 1

            out['s2s2r'][source][sink] = rel

    if __debug__:
        flat_out, flat_base, flat_extend = tuple([
            len(flatten_source2sink2rel(e)) for e in [out['s2s2r'], base, extend]
        ])
        assert flat_base <= flat_out, {'base': flat_base, 'out': flat_out}
        assert flat_extend == out['confirmed'] + would_add + would_change

    return out