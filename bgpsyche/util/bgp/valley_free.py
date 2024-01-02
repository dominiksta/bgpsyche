import typing as t

from bgpsyche.util.path_finding.path_is_in_graph import path_is_in_graph

from .relationship import RelationshipKind, Source2Sink2Rel


def _rels_left_is_smaller(a: RelationshipKind, b: RelationshipKind) -> bool:
    rels_numeric: t.Dict[RelationshipKind, int] = {
        'c2p': 1,
        'p2p': 0,
        'p2c': -1,
    }
    return rels_numeric[a] < rels_numeric[b]

def path_is_valley_free(rels: Source2Sink2Rel, path: t.List[int]) -> t.Optional[bool]:
    """Check that the given path is valley free.

    Explanation from "AS relationships, customer cones, and validation", Luckie et
    al., 10.1145/2504730.2504735:

    "...each path consists of an uphill segment of zero or more c2p or sibling links,
    zero or one p2p links at the top of the path, followed by a downhill segment of
    zero or more p2c or sibling links"
    """

    if not path_is_in_graph(t.cast(t.Dict[int, t.Iterable[int]], rels), path):
        return None

    relationships: t.List[RelationshipKind] = [
        rels[path[i]][path[i+1]] for i in range(0, len(path) - 1)
    ]

    for i in range(0, len(relationships) - 1):
        if _rels_left_is_smaller(relationships[i], relationships[i+1]):
            return False

    if relationships.count('p2p') > 1: return False

    return True
