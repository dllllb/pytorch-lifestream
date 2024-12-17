import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch import load
from pytorch_lightning.loggers import TensorBoardLogger

from ptls.fedcore_compression.fc_utils import fedcore_fit, extract_loss, get_experimental_setup

logger = logging.getLogger(__name__)

def fold_fit_test(conf: DictConfig, fold_id: int):
    """
    Runs training and test inference at once on a data fold to lessen computation costs.

    Parameters:
        conf: configuration with specified:
        * `pretrained` - pretrained encodef path
        * `setup` - experimental setup for compression
        * `trainer` - pl.Trainer settings
        * `embedding_validation_results.output_path` - the output path
        fold_id: fold number
    """
    conf['fold_id'] = fold_id

    # model instantiation
    model = hydra.utils.instantiate(conf.pl_module)
    model.seq_encoder.is_reduce_sequence = True
    if 'pretrained' in conf and conf.pretrained:
        model.seq_encoder = load(conf.pretrained)
    loss = extract_loss(model, conf)

    # data module instantiation
    dm = hydra.utils.instantiate(conf.data_module)

    # experiment setup
    experiment_setup, setup_name = get_experimental_setup(conf.get('setup', None))
    model_name = conf.get('model_name', '')
    save_path = f'composition_results/{setup_name}{f"_{model_name}" if model_name else ""}'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    experiment_setup['output_folder'] = save_path
    if not 'common' in experiment_setup:
        experiment_setup['common'] = {}
    experiment_setup['common']['batch_limit'] = conf.get('limit_train_batches', None)
    experiment_setup['common']['calib_batch_limit'] = conf.get('limit_valid_batches', None)
    experiment_setup['need_fedot_pretrain'] = conf.get('need_fedot_pretrain', False)
    experiment_setup['need_evo_opt'] = conf.get('need_evo_opt', False)
    experiment_setup['distributed_compression'] = conf.get('distributed_compression', False)
    
    # training
    fedcore_compressor = fedcore_fit(model, dm, experiment_setup, loss)

    # init trainer for prediction
    _trainer_params = conf.trainer
    _trainer_params_additional = {}
    if 'logger_name' in conf:
        _trainer_params_additional['logger'] = TensorBoardLogger(
            save_dir='lightning_logs',
            name=conf.get('logger_name'),
        )
    trainer = pl.Trainer(**_trainer_params, **_trainer_params_additional)

    trainer.test(model=fedcore_compressor.optimised_model, 
                 dataloaders=dm.test_dataloader(), 
                 ckpt_path=None, 
                 verbose=False)
    test_metrics = {
        name: value.item() for name, value in trainer.logged_metrics.items() if name.startswith('test')
    }
    print(', '.join([f' {name}: {v:.4f}' for name, v in test_metrics.items()]))

    return {
        "fold_id": fold_id,
        "model_name": conf.embedding_validation_results.model_name,
        "feature_name": conf.embedding_validation_results.feature_name,
        'scores_test': test_metrics,
    }


@hydra.main(version_base=None)
def main(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    fold_list = hydra.utils.instantiate(conf.fold_list)

    results = []
    for fold_id in fold_list:
        logger.info(f'==== Fold [{fold_id}/{len(fold_list)}] fit-test start ====')
        result = fold_fit_test(conf, fold_id)
        results.append(result)

    stats_file = conf.embedding_validation_results.output_path
    if stats_file is not None:
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Embeddings validation scores saved to "{stats_file}"')

    for name in results[0]['scores_test'].keys():
        test_scores = np.array([x['scores_test'][name] for x in results])
        print(f'Test  {name:10}: {test_scores.mean():.3f} [{", ".join(f"{t:.3f}" for t in test_scores)}]')


if __name__ == '__main__':
    main()
