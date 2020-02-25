import json
from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

import pandas as pd

pd.options.display.float_format = "{:.4f}".format
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
pd.options.display.width = None


# history_path = '../history/ml_vector_quality-color-0.0001-1.4-1024.json'
# history_path = '../history/ml_vector_quality-color-0.0001-0.9-256.json'
# history_path = '../history/ml_vector_quality-color-0.0001-0.4-256.json'  # BEST
# history_path = '../history/ml_vector_quality-color-0.0001-0.15-256.json'

# history_path = '../history/ml_vector_quality-size-0.0001-0.6-196.json'
history_path = '../history/ml_vector_quality-size-0.0001-0.4-196.json'  # BEST
# history_path = '../history/ml_vector_quality-size-0.0001-0.2-196.json'


if __name__ == '__main__':
    with open(history_path) as f:
        pd_report = json_normalize(json.load(f))
    pd_report = pd_report.set_index('current_epoch')

    print(pd_report)

    sup_losses = pd_report.columns[pd_report.columns.str.startswith('sup_score.loss_')]
    pd_report[sup_losses].plot(
        figsize=(12, 18), grid=True,
        title='Supervised losses',
    )
    random_level = 2 * 0.5**3 / 3
    plt.axhline(random_level, color='red', linestyle='--')
    plt.axhline(3 * random_level, color='red', linestyle='--', label='constant predict baseline')
    plt.axhline(0.011887093215057301, color='blue', linestyle='--', label='supervised 40 epoch baseline')
    plt.axhline(0.006410095581991805, color='green', linestyle='--', label='supervised 80 epoch baseline')
    plt.legend()
    plt.show()

    pd_report['dist_mean'] = pd_report['ml_score.dist_pos_mean'] - pd_report['ml_score.dist_neg_mean']
    pd_report[['ml_score.dist_neg_mean', 'ml_score.dist_pos_mean', 'dist_mean']].plot(
        figsize=(12, 6), grid=True, secondary_y='dist_mean',
        title='Metric learning mean distances',
    )
    plt.show()

    pd_report['dist_min_max'] = pd_report['ml_score.dist_pos_max'] - pd_report['ml_score.dist_neg_min']
    pd_report[['ml_score.dist_neg_min', 'ml_score.dist_pos_max', 'dist_min_max']].plot(
        figsize=(12, 6), grid=True, secondary_y='dist_min_max',
        title='Metric learning min-max distances',
    )
    plt.show()
