import logging

from scenario_tinkoff import parse_args, main

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-15s : %(message)s')

    main(parse_args(None))
