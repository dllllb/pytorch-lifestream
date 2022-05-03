import json
import logging
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger

from ptls.data_load.data_module.cls_data_module import ClsDataModuleTrain
import pytorch_lightning as pl

from ptls.seq_to_target import SequenceToTarget
from ptls.util import get_conf, get_cls

logger = logging.getLogger(__name__)


def fold_fit_test(conf, fold_id, pretrained_module=None):
    pretrained_encoder = None if pretrained_module is None else pretrained_module.seq_encoder
    model = SequenceToTarget(conf['params'], pretrained_encoder)
    dm = ClsDataModuleTrain(conf['data_module'], model, fold_id)

    _trainer_params = conf['trainer']
    if 'logger_name' in conf:
        _trainer_params['logger'] = TensorBoardLogger(
            save_dir='lightning_logs',
            name=conf.get('logger_name'),
        )
    trainer = pl.Trainer(**_trainer_params)
    trainer.fit(model, dm)

    if 'model_path' in conf:
        trainer.save_checkpoint(conf['model_path'], weights_only=True)
        logger.info(f'Model weights saved to "{conf.model_path}"')

    valid_metrics = {name: float(mf.compute().item()) for name, mf in model.valid_metrics.items()}
    trainer.test(test_dataloaders=dm.test_dataloader(), ckpt_path=None, verbose=False)
    test_metrics = {name: float(mf.compute().item()) for name, mf in model.test_metrics.items()}

    print(', '.join([f'valid_{name}: {v:.4f}' for name, v in valid_metrics.items()]))
    print(', '.join([f' test_{name}: {v:.4f}' for name, v in test_metrics.items()]))

    return {
        "fold_id": fold_id,
        "model_name": conf['embedding_validation_results.model_name'],
        "feature_name": conf['embedding_validation_results.feature_name'],
        'scores_valid': valid_metrics,
        'scores_test': test_metrics,
    }


def main(args=None):
    conf = get_conf(args)

    if 'seed_everything' in conf:
        pl.seed_everything(conf['seed_everything'])

    if conf['data_module.setup.split_by'] == 'embeddings_validation':
        with open(conf['data_module.setup.fold_info'], 'r') as f:
            fold_info = json.load(f)
        fold_list = [k for k in sorted(fold_info.keys()) if not k.startswith('_')]
    else:
        raise NotImplementedError(f'Only `embeddings_validation` split supported,'
                                  f'found "{conf.data_module.setup.split_by}"')

    if conf['params.encoder_type'] == 'pretrained':
        pre_conf = conf['params.pretrained']
        cls = get_cls(pre_conf['pl_module_class'])
        pretrained_module = cls.load_from_checkpoint(pre_conf['model_path'])
        pretrained_module.seq_encoder.is_reduce_sequence = True
    else:
        pretrained_module = None

    results = []
    for fold_id in fold_list:
        logger.info(f'==== Fold [{fold_id}] fit-test start ====')
        result = fold_fit_test(conf, fold_id, pretrained_module)
        results.append(result)

    stats_file = conf['embedding_validation_results.output_path']
    if stats_file is not None:
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Embeddings validation scores saved to "{stats_file}"')

    for name in results[0]['scores_valid'].keys():
        valid_scores = np.array([x['scores_valid'][name]for x in results])
        test_scores = np.array([x['scores_test'][name] for x in results])
        print(f'Valid {name:10}: {valid_scores.mean():.3f} [{", ".join(f"{t:.3f}" for t in valid_scores)}], '
              f'Test  {name:10}: {test_scores.mean():.3f} [{", ".join(f"{t:.3f}" for t in test_scores)}]')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')
    logging.getLogger("lightning").setLevel(logging.INFO)

    main()
