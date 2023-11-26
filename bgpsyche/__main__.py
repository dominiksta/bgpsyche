import argparse
from datetime import datetime
import logging

from bgpsyche import logging_config
from bgpsyche.service.bgp_markov_chain import markov_chain_from_ripe_ris

_LOG = logging.getLogger(__name__)

_parser = argparse.ArgumentParser()

_parser.add_argument(
    '--test', required=False
)

_args = _parser.parse_args()

if _args.test:
    markov_chain_from_ripe_ris(datetime.fromisoformat('2023-05-01T00:00'))