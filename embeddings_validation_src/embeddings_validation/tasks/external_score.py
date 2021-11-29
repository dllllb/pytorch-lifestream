import json
import logging
import os

import luigi

from embeddings_validation.config import Config


class ExternalScore(luigi.Task):
    conf = luigi.Parameter()
    name = luigi.Parameter()
    external_path = luigi.Parameter()

    def output(self):
        conf = Config.read_file(self.conf)
        path = os.path.join(conf.work_dir, 'external', self.name, 'scores.json')
        return luigi.LocalTarget(path)

    def run(self):
        conf = Config.read_file(self.conf)
        on_error = conf.error_handling

        try:
            path = os.path.join(conf.root_path, self.external_path)
            with open(path, 'r') as f:
                external_scores = json.load(f)
        except BaseException:
            if on_error == conf.ON_ERROR_SKIP:
                external_scores = None
                logging.getLogger('luigi-interface').exception('Fail', stack_info=True)
            elif on_error == conf.ON_ERROR_FAIL:
                raise
            else:
                raise AssertionError(f'Unknown error_handling: "{on_error}"')

        with self.output().open('w') as f:
            external_scores = [] if external_scores is None else external_scores
            json.dump(external_scores, f, indent=2)
