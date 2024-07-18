import logging
import typing as t
from datetime import datetime

from bgpsyche.service.ext import ripe_ris
from bgpsyche.service.ext.caida_asrel import get_caida_asrel
from bgpsyche.util.bgp.valley_free import path_is_valley_free

_LOG = logging.getLogger(__name__)

def _research_valley_free_how_many():
    is_valley_free: t.List[t.Optional[bool]] = []
    dt = datetime.fromisoformat('2023-05-01T00:00')

    for path_meta in ripe_ris.iter_paths(
            dt, eliminate_path_prepending=True,
    ):
        path = path_meta['path']
        is_valley_free.append(path_is_valley_free(get_caida_asrel(dt), path))

    _LOG.info({
        'true': is_valley_free.count(True),
        'false': is_valley_free.count(False),
        'unknown': is_valley_free.count(None),
    })



if __name__ == '__main__': _research_valley_free_how_many()