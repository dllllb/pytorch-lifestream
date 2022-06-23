import hydra
import json
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from ptls.lightning_modules.rtd_module import RtdModule
import pytorch_lightning as pl


logger = logging.getLogger(__name__)


def fold_fit_test(conf, pl_module, fold_id):
    if pl_module:
        model = hydra.utils.instantiate(conf.pl_module, seq_encoder=pl_module.seq_encoder)
    else:
        model = hydra.utils.instantiate(conf.pl_module)
    dm = hydra.utils.instantiate(conf.data_module, pl_module=model, fold_id=fold_id)

    _trainer_params = conf.trainer
    _trainer_params_additional = {}
    if 'logger_name' in conf:
        _trainer_params_additional['logger'] = TensorBoardLogger(
            save_dir='lightning_logs',
            name=conf.get('logger_name'),
        )
    trainer = pl.Trainer(**_trainer_params, **_trainer_params_additional)
    trainer.fit(model, dm)

    if 'model_path' in conf:
        trainer.save_checkpoint(conf.model_path, weights_only=True)
        logger.info(f'Model weights saved to "{conf.model_path}"')

    valid_metrics = {name: float(mf.compute().item()) for name, mf in model.valid_metrics.items()}
    trainer.test(test_dataloaders=dm.test_dataloader(), ckpt_path=None, verbose=False)
    test_metrics = {name: float(mf.compute().item()) for name, mf in model.test_metrics.items()}

    print(', '.join([f'valid_{name}: {v:.4f}' for name, v in valid_metrics.items()]))
    print(', '.join([f' test_{name}: {v:.4f}' for name, v in test_metrics.items()]))

    return {
        "fold_id": fold_id,
        "model_name": conf.embedding_validation_results.model_name,
        "feature_name": conf.embedding_validation_results.feature_name,
        'scores_valid': valid_metrics,
        'scores_test': test_metrics,
    }

@hydra.main()
def main(conf: DictConfig):
    OmegaConf.set_struct(conf, False)

    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    if conf.data_module.setup.split_by == 'embeddings_validation':
        with open(conf.data_module.setup.fold_info, 'r') as f:
            fold_info = json.load(f)
        fold_list = [k for k in sorted(fold_info.keys()) if not k.startswith('_')]
    else:
        raise NotImplementedError(f'Only `embeddings_validation` split supported,'
                                  f'found "{conf.data_module.setup.split_by}"')
        
    pretrained_encoder_path = conf.get('pretrained_encoder_path', None)
    if pretrained_encoder_path:
        # pl_module_cls = hydra.utils.instantiate(conf.pretrained_module_cls)
        # pl_module = pl_module_cls.load_from_checkpoint(pretrained_encoder_path)
        pl_module = hydra.utils.instantiate(conf.pretrained_module)
        state_dict = torch.load(pretrained_encoder_path)['state_dict']
        if not isinstance(pl_module, RtdModule):
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_head')}
        pl_module.load_state_dict(state_dict)
        pl_module.seq_encoder.is_reduce_sequence = True
    else:
        pl_module = None
    use_pretrained_path = conf.get('use_pretrained_path', None)

    results = []
    for fold_id in fold_list:
        logger.info(f'==== Fold [{fold_id}] fit-test start ====')
        result = fold_fit_test(conf, pl_module, fold_id)
        results.append(result)

    stats_file = conf.embedding_validation_results.output_path
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
