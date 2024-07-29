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