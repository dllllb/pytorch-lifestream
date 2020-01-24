import logging
import numpy as np

from scenario_tinkoff import parse_args, main

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-15s : %(message)s')

    np.set_printoptions(linewidth=160)

    main(parse_args(None))
