import argparse
import logging
from pprint import pprint

from bgpsyche.logging_config import logging_setup
from bgpsyche.service.ext import peeringdb
from bgpsyche.stage1_candidates import get_path_candidates
from bgpsyche.stage2_enrich.enrich import enrich_path

logging_setup()
_LOG = logging.getLogger(__name__)

_parser = argparse.ArgumentParser()

_parser.add_argument(
    '--stage', choices=[
        '01_get_path_candidates',
        '02_enrich_path_candidates',
    ],
    required=False
)

_parser.add_argument('--sync-peeringdb', action='store_true', required=False)

_args = _parser.parse_args()

if _args.sync_peeringdb: peeringdb.Client.sync()

if _args.stage == '01_get_path_candidates':
    # 23673 23764 4134 4538 23910 24371
    candidates = get_path_candidates(23673, 24371)
    print([23673, 23764, 4134, 4538, 23910, 24371] in candidates['by_length'][6])
    print(len(candidates['candidates']))


    # 14840 →  32098 ↘  13999 ↘ 265620
    candidates = get_path_candidates(14840, 265620)
    pprint(candidates['candidates'][:10])

    print([14840, 32098, 13999, 265620] in candidates['by_length'][4])
    print(candidates['by_length'].keys())

elif _args.stage == '02_enrich_path_candidates':
    candidates = get_path_candidates(14840, 265620)['candidates']
    print(len(candidates))

    for path in candidates[:50]:
        pprint({ 'path': path, 'features': enrich_path(path) })
