import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split 
import sklearn.metrics as METRICS
from sklearn.pipeline import make_pipeline
from utils import load_pkl, load_json, save_json, save_pkl


@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    type_ = conf.type_
    folder = conf.emb_path
    conf.mles_embeddings.read_params.file_name = "composition_results/{type_}/scores"

    path = Path(folder, type_, 'scores')
    labels = load_pkl('embeddings_validation.work/folds/target_test.pickle').df
    labels = labels.set_index('app_id')
    metrics = hydra.utils.instantiate(conf.metrics)

    # print(models)
    for i, ckpt_embs in enumerate(os.listdir(path)):
        if not ckpt_embs.endswith('.csv'): continue
        model_embs = Path(path, ckpt_embs)
        embs = pd.read_csv(model_embs)
        ids = embs['app_id']
        labels_ = labels.loc[ids, 'flag']
        xtr, xte, ytr, yte = train_test_split(embs, labels_, test_size=0.3)
        for model_name, model_dict in conf.models.items():
            if not model_dict.enabled: continue
            ppl = make_pipeline(*[hydra.utils.instantiate(model_dict.preprocessing[i]) 
                                    for i in range(len(model_dict.preprocessing))],
                hydra.utils.instantiate(model_dict.model)
            ).fit(xtr, ytr)
            
            for name, metric in metrics.items():
                if not metric['enabled']: continue
                ypred = ppl.predict(xte)
                value = getattr(METRICS, metric['score_func'])(yte, ypred)
                with open(Path(path, 'eval.txt'), 'at+') as file:
                    print(f'#{i} {model_name} metric: {name} value: {value}', file=file)

if __name__ == '__main__':
    main()
