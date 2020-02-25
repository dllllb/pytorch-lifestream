import logging
import os

import numpy as np
from tools import logging_base_config
from tools.dataset import create_supervised, save_dataset, create_metric_learning

logger = logging.getLogger(__name__)

SIZES_SUPERVISED = [
    ('train', 4000, [], None),
    ('valid', 4000, [], None),
    #
    ('train_gen1', 2000, [], None),
    ('train_gen2', 2000, [], None),
    ('valid_gen1', 2000, [], None),
    ('valid_gen2', 2000, [], None),
    #
    ('train_medium_size', 2000, [], {'size': [0.5]}),
    ('valid_medium_size', 2000, [], {'size': [0.5]}),
    #
    ('train_gen_blue', 2000, [], {'color': np.array([0.0, 0.0, 1.0], dtype=np.float32)}),
    ('train_gen_blue_on_white', 2000, [], {
        'color': np.array([0.0, 0.0, 1.0], dtype=np.float32),
        'bg_color': np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }),
    ('train_gen_red', 2000, [], {'color': np.array([1.0, 0.0, 0.0], dtype=np.float32)}),
    ('valid_gen_blue', 2000, [], {'color': np.array([0.0, 0.0, 1.0], dtype=np.float32)}),
    ('valid_gen_red', 2000, [], {'color': np.array([1.0, 0.0, 0.0], dtype=np.float32)}),
]
SIZES_METRIC_LEARNING = [4000]

ML_SAMPLE_COUNT = 4
CLASS_PARAMS = [
    ['color'],
    ['bg_color'],
    ['size'],
    ['n_vertex'],
    ['size', 'n_vertex', 'bg_color'],
]


if __name__ == '__main__':
    logging_base_config()

    for name, size, keep_params, params in SIZES_SUPERVISED:
        file_name = f'../datasets/supervised-{name}-{size}.p'
        if os.path.isfile(file_name):
            logger.info(f'Skipped "{file_name}"')
        else:
            data = create_supervised(size, {'keep_params': keep_params, 'params': params})
            save_dataset(data, file_name)

    for size in SIZES_METRIC_LEARNING:
        for class_param in CLASS_PARAMS:
            class_param_str = "_".join(sorted(class_param))
            file_name = f'../datasets/metric_learning-{size}-{ML_SAMPLE_COUNT}-{class_param_str}.p'
            if os.path.isfile(file_name):
                logger.info(f'Skipped "{file_name}"')
            else:
                data = create_metric_learning(size, ML_SAMPLE_COUNT, {'keep_params': class_param})
                save_dataset(data, file_name)
