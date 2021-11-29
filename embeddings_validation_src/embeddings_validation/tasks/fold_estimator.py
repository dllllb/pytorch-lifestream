import datetime
import json
import logging
import math
import os

import luigi

from embeddings_validation.config import Config
from embeddings_validation.file_reader import TargetFile
from embeddings_validation.metrics import Metrics
from embeddings_validation import cls_loader
from embeddings_validation.tasks.fold_splitter import FoldSplitter
from embeddings_validation.x_transformer import XTransformer


class FoldEstimator(luigi.Task):
    conf = luigi.Parameter()
    model_name = luigi.Parameter()
    feature_name = luigi.Parameter()
    fold_id = luigi.IntParameter()

    total_cpu_count = luigi.IntParameter()

    def requires(self):
        return FoldSplitter(conf=self.conf)

    def output(self):
        conf = Config.read_file(self.conf)
        fold_name = '__'.join([
            f'm_{self.model_name}',
            f'f_{self.feature_name}',
            f'i_{self.fold_id}'
        ])
        path = os.path.join(conf.work_dir, fold_name, 'results.json')
        return luigi.LocalTarget(path)

    @property
    def resources(self):
        conf = Config.read_file(self.conf)

        cpu_count = conf.models[self.model_name]['cpu_count']
        return {'cpu': math.ceil(cpu_count / self.total_cpu_count * 100) / 100}

    def run(self):
        conf = Config.read_file(self.conf)

        x_transf = XTransformer(conf, self.feature_name, conf.models[self.model_name]['preprocessing'])
        conf_model = conf.models[self.model_name]
        model = cls_loader.create(conf_model['cls_name'], conf_model['params'])
        scorer = Metrics(conf)
        on_error = conf.error_handling

        results = {
            'fold_id': self.fold_id,
            'model_name': self.model_name,
            'feature_name': self.feature_name,
        }

        # folds with train-valid-test ids and targets
        with self.input().open('r') as f:
            folds = json.load(f)
        current_fold = folds[str(self.fold_id)]

        try:
            target_train = TargetFile.load(current_fold['train']['path'])
            target_train = self.apply_target_options(target_train)

            _start = datetime.datetime.now()
            X_train = x_transf.fit_transform(target_train)
            train_fit_transform_time = datetime.datetime.now() - _start

            _start = datetime.datetime.now()
            model.fit(X_train, target_train.target_values)
            train_time = datetime.datetime.now() - _start

            if scorer.is_check_train:
                results['scores_train'] = scorer.score(model, X_train, target_train.target_values)
            results['scores_valid'] = self.score_data(current_fold['valid']['path'], x_transf, model, scorer)
            if current_fold['test'] is not None:
                results['scores_test'] = self.score_data(current_fold['test']['path'], x_transf, model, scorer)

            results['process_info'] = {
                'feature_fit_info': x_transf.get_feature_fit_info(),
                'feature_load_time': x_transf.load_time.seconds,
                'train_fit_transform_time': train_fit_transform_time.seconds,
                'train_time': train_time.seconds,
            }

        except BaseException:
            if on_error == conf.ON_ERROR_SKIP:
                results = None
                logging.getLogger('luigi-interface').exception('Fail', stack_info=True)
            elif on_error == conf.ON_ERROR_FAIL:
                raise
            else:
                raise AssertionError(f'Unknown error_handling: "{on_error}"')

        with self.output().open('w') as f:
            results = [] if results is None else [results]
            json.dump(results, f, indent=2)

    def score_data(self, target_path, x_transf, model, scorer):
        target_data = TargetFile.load(target_path)

        _start = datetime.datetime.now()
        X_data = x_transf.transform(target_data)
        feature_transform_time = datetime.datetime.now() - _start

        score = scorer.score(model, X_data, target_data.target_values)
        score['feature_transform_time'] = feature_transform_time.seconds
        return score

    def apply_target_options(self, df_target):
        conf = Config.read_file(self.conf)
        target_options = conf.features[self.feature_name]['target_options']
        if 'labeled_amount' in target_options:
            df_target = self.reduce_labeled_amount(
                df_target,
                target_options['labeled_amount']
            )
        return df_target

    def reduce_labeled_amount(self, df_target, labeled_amount):
        conf = Config.read_file(self.conf)
        row_order_shuffle_seed = conf['split'].get('row_order_shuffle_seed', None)
        if row_order_shuffle_seed is None:
            logging.warning('`row_order_shuffle_seed` not set. Row order may be incorrect')

        if type(labeled_amount) is float and labeled_amount <= 1.0:
            n = int(len(df_target) * labeled_amount)
        elif type(labeled_amount) is int:
            n = labeled_amount
        else:
            raise AttributeError(f'wrong format of labeled_amount ({labeled_amount}), type: {type(labeled_amount)}')

        new_target = df_target.clone_schema()
        new_target.df = df_target.df.iloc[:n]
        return new_target
