from datetime import datetime

from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.service.bgp_graph import as_graphs_from_paths
from bgpsyche.service.ext import ripe_ris


as_graph_ris = PickleFileCache(
    'as_graphs',
    lambda: as_graphs_from_paths(ripe_ris.iter_paths(
        datetime.fromisoformat('2023-05-01T00:00')
    ))[0]
)
