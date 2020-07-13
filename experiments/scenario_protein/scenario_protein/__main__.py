import argparse
import logging
import numpy as np
import torch

if __name__ == '__main__':
    import sys
    sys.path.append('../../')

    # reproducibility
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

from scenario_protein import compare_approaches, fit_target, fit_finetuning  # , pseudo_labeling

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    for module in [
        compare_approaches,
        fit_target,
        fit_finetuning,
        # pseudo_labeling,
        ]:
        sub_parser = subparsers.add_parser(module.__name__.split('.')[-1])
        sub_parser.set_defaults(func=module.main)
        module.prepare_parser(sub_parser)

    config, _ = parser.parse_known_args(args)
    return vars(config)


if __name__ == '__main__':
    conf = parse_args()
    if 'func' not in conf:
        print('Choose scenario. Use --help for more information')
        exit(-1)

    conf['func'](conf)
