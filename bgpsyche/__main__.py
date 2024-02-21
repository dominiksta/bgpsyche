import argparse
import logging
from os import system
from pprint import pprint

from bgpsyche.logging_config import logging_setup
from bgpsyche.service.ext import peeringdb
from bgpsyche.stage1_candidates import get_path_candidates
from bgpsyche.stage2_enrich.enrich import enrich_path

logging_setup()
_LOG = logging.getLogger(__name__)

_parser = argparse.ArgumentParser()

_parser.add_argument('action', choices=[
    'test_stage1',
    'test_stage2',
    'tensorboard',
    'peeringdb_sync',
])

_args = _parser.parse_args()

if _args.action == 'peeringdb_sync': peeringdb.Client.sync()

elif _args.action == 'test_stage1':
    # 23673 23764 4134 4538 23910 24371
    candidates = get_path_candidates(23673, 24371)
    print(len(candidates))


    # 14840 →  32098 ↘  13999 ↘ 265620
    candidates = get_path_candidates(14840, 265620)
    pprint(candidates[:10])


elif _args.action == 'test_stage2':
    candidates = get_path_candidates(14840, 265620)
    print(len(candidates))

    for path in candidates[:50]:
        pprint({ 'path': path, 'features': enrich_path(path) })


elif _args.action == 'tensorboard':
    system('tensorboard --load_fast=false --logdir=./bgpsyche/data/tensorboard/')