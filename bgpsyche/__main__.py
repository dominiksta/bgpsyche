import argparse
import logging
from os import system
from pprint import pprint

from bgpsyche.logging_config import logging_setup
from bgpsyche.service.ext import peeringdb
from bgpsyche.stage1_candidates import get_path_candidates
from bgpsyche.stage2_enrich.enrich import enrich_path
from bgpsyche.stage3_rank.tensorboard import make_tensorboard_writer

logging_setup()
_LOG = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='command', required=True)

    peeringdb_sync = subparsers.add_parser('peeringdb_sync')
    tensorboard    = subparsers.add_parser('tensorboard')

    train_and_eval = subparsers.add_parser('train_and_eval')
    train_and_eval.add_argument('name', help='Name (for Tensorboard)')

    args = parser.parse_args()

    _LOG.info(f'Started from main with command <<{args.command}>>')
    _LOG.info(f'{args}')

    if args.command == 'peeringdb_sync':
        peeringdb.Client.sync()

    elif args.command == 'train_and_eval':
        make_tensorboard_writer(args.name)
        # HACK: this is kinda cursed, but we have to define the prediction
        # function globally in the real_life_eval module, because pickling a
        # function object and passing it to workers is not possible. but
        # defining the prediction function requires training. there is probably
        # a better way of doing this but this inline import works ok for now.
        from bgpsyche.stage3_rank.real_life_eval import real_life_eval_model
        real_life_eval_model()

    elif args.command == 'tensorboard':
        system(
            'tensorboard ' +
            '--bind_all ' +
            '--load_fast=false ' +
            '--logdir=./bgpsyche/data/tensorboard/'
        )

if __name__ == '__main__': main()