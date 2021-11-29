import datetime

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import make_scorer

from embeddings_validation import cls_loader


class Metrics:
    def __init__(self, conf):
        self.is_check_train = conf['report']['is_check_train']

        self.metric_list = {k: self.get_scorer(**v) for k, v in conf.metrics.items()}

    def get_scorer(self, score_func, scorer_params):
        score_f = cls_loader.get_cls(score_func)

        return make_scorer(score_f, **scorer_params)

    def score(self, model, X, y):
        results = {
            'cnt_samples': X.shape[0],
            'cnt_features': X.shape[1],
        }

        for name, scorer in self.metric_list.items():
            _start = datetime.datetime.now()
            results[name] = scorer(model, X, y)
            _duration = datetime.datetime.now() - _start
            results[f'{name}_score_time'] = _duration.seconds

        return results
