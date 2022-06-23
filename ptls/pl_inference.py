import logging

import glob
import torch
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ChainDataset
from torch.utils.data.dataloader import DataLoader

from ptls.data_load import IterableChain, padded_collate, IterableAugmentations
from ptls.data_load.augmentations.seq_len_limit import SeqLenLimit
from ptls.data_load.iterable_processing.category_size_clip import CategorySizeClip
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load.iterable_processing.target_extractor import TargetExtractor
from ptls.data_load.parquet_dataset import ParquetDataset, ParquetFiles
from ptls.lightning_modules.rtd_module import RtdModule
from ptls.metric_learn.inference_tools import save_scores
from ptls.train import score_model
from ptls.util import get_conf, get_cls

logger = logging.getLogger(__name__)


def create_inference_dataloader(conf, pl_module):
    """This is inference dataloader for `experiments`
    """
    post_processing = IterableChain(
        TargetExtractor(target_col=conf.col_id),
        FeatureFilter(keep_feature_names=pl_module.seq_encoder.category_names),
        CategorySizeClip(pl_module.seq_encoder.category_max_size),
        IterableAugmentations(
            SeqLenLimit(**conf.SeqLenLimit),
        )
    )
    l_dataset = [
        ParquetDataset(
            ParquetFiles(path).data_files,
            post_processing=post_processing,
            shuffle_files=False,
        ) for path in conf.dataset_files]
    dataset = ChainDataset(l_dataset)
    return DataLoader(
        dataset=dataset,
        collate_fn=padded_collate,
        shuffle=False,
        num_workers=conf.loader.num_workers,
        batch_size=conf.loader.batch_size,
    )


@hydra.main()
def main(conf: DictConfig):
    OmegaConf.set_struct(conf, False)

    if 'torch_multiprocessing_sharing_strategy' in conf.inference_dataloader:
        torch.multiprocessing.set_sharing_strategy(
            conf.inference_dataloader.torch_multiprocessing_sharing_strategy
        )

    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    model = hydra.utils.instantiate(conf.pl_module)

    if not conf.get('random_model', False):
        # model = model.load_from_checkpoint(conf.model_path)
        state_dict = torch.load(conf.model_path)['state_dict']
        if not isinstance(model, RtdModule):
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_head')}
        model.load_state_dict(state_dict)

    model.seq_encoder.is_reduce_sequence = True

    dl = create_inference_dataloader(conf.inference_dataloader, model)

    pred, ids = score_model(model, dl, conf.device)

    df_scores_cols = [f'v{i:003d}' for i in range(pred.shape[1])]
    col_id = conf.inference_dataloader.col_id
    df_scores = pd.concat([
        pd.DataFrame({col_id: ids}),
        pd.DataFrame(pred, columns=df_scores_cols),
        ], axis=1)
    logger.info(f'df_scores examples: {df_scores.shape}:')

    save_scores(df_scores, None, conf.output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')
    logging.getLogger("lightning").setLevel(logging.INFO)

    main()
