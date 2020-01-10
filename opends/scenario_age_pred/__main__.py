import argparse
import logging

from scenario_age_pred import compare_approaches

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    sub_parser = subparsers.add_parser('compare_approaches')
    sub_parser.set_defaults(func=compare_approaches.main)
    compare_approaches.prepare_parser(sub_parser)

    config = parser.parse_args(args)
    return vars(config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')

    conf = parse_args()
    if 'func' not in conf:
        logger.error('Choose scenario. Use --help for more information')
        exit(-1)

    conf['func'](conf)
