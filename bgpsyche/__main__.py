import argparse
from datetime import datetime
import functools
import logging
from pprint import pprint

from bgpsyche import logging_config
from bgpsyche.stage1_candidates import get_path_candidates
from bgpsyche.stage1_candidates.get_candidates import flatten_candidates
from bgpsyche.stage2_enrich.enrich import enrich_path

_LOG = logging.getLogger(__name__)

_parser = argparse.ArgumentParser()

_parser.add_argument(
    '--stage', choices=[
        '01_get_path_candidates',
        '02_enrich_path_candidates',
    ],
    required=False
)

_args = _parser.parse_args()

if _args.stage == '01_get_path_candidates':
    # 23673 23764 4134 4538 23910 24371
    candidates = get_path_candidates(23673, 24371)
    print([23673, 23764, 4134, 4538, 23910, 24371] in candidates['all'][6])
    print(len(candidates['all'][6]))


    # 14840 →  32098 ↘  13999 ↘ 265620
    candidates = get_path_candidates(14840, 265620)
    pprint(candidates['shortest'])

    print([14840, 32098, 13999, 265620] in candidates['all'][4])
    print(candidates['all'].keys())

elif _args.stage == '02_enrich_path_candidates':
    # 23673 23764 4134 4538 23910 24371
    candidates = flatten_candidates(get_path_candidates(23673, 24371))

    pprint(enrich_path(candidates[0]))
