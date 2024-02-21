import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np

from itertools import chain
from ptls.data_load.padded_batch import PaddedBatch


class InferenceModule(pl.LightningModule):
    def __init__(self, model, pandas_output=True, drop_seq_features=True, model_out_name='out'):
        super().__init__()

        self.model = model
        self.pandas_output = pandas_output
        self.drop_seq_features = drop_seq_features
        self.model_out_name = model_out_name

    def forward(self, x: PaddedBatch):
        out = self.model(x)
        if self.drop_seq_features:
            x = x.drop_seq_features()
            x[self.model_out_name] = out
        else:
            x.payload[self.model_out_name] = out
        if self.pandas_output:
            return self.to_pandas(x)
        return x

    def to_pandas(self, x):
        is_reduced = None
        scalar_features, seq_features, expand_features = {}, {}, {}
        df_scalar, df_seq, df_expand, out_df = None, None, None, None
        len_mask = None

        x_ = x
        if type(x_) is PaddedBatch:
            len_mask = x_.seq_len_mask.bool().cpu().numpy()
            x_ = x_.payload
        is_reduced = (type(x_[self.model_out_name]) is not PaddedBatch)
        for k, v in x_.items():
            if type(v) is PaddedBatch:
                len_mask = v.seq_len_mask.bool().cpu().numpy()
                v = v.payload
            if type(v) is torch.Tensor:
                v = v.cpu().numpy()
            if type(v) is list or len(v.shape) == 1:
                scalar_features[k] = v
            elif k.startswith('target'):
                scalar_features[k] = v
            elif len(v.shape) == 3:
                expand_features[k] = v
            elif k == self.model_out_name and len(v.shape) == 2:
                expand_features[k] = v
            elif len(v.shape) == 2:
                seq_features[k] = v

        if is_reduced:
            df_scalar, df_seq, df_expand = self.to_pandas_record(x, expand_features, scalar_features, seq_features, len_mask)
        else:
            df_scalar, df_seq, df_expand = self.to_pandas_sequence(x, expand_features, scalar_features, seq_features, len_mask)

        out_df = df_scalar
        if df_seq:
            df_seq = pd.concat(df_seq, axis = 1)
            out_df = pd.concat([df_scalar, df_seq], axis = 1)
        if df_expand:
            df_expand = pd.concat(df_expand, axis = 0).reset_index(drop=True)
            out_df = pd.concat([out_df.reset_index(drop=True), df_expand], axis = 1)

        return out_df

    @staticmethod
    def to_pandas_record(x, expand_features, scalar_features, seq_features, len_mask):
        dataframes_scalar = []
        for k, v in scalar_features.items():
            dataframes_scalar.append(pd.DataFrame(v, columns=[k]))
        dataframes_scalar = pd.concat(dataframes_scalar, axis = 1)

        dataframes_seq = []
        for k, v in seq_features.items():
            data_lst = [usr[len_mask[i]] for i, usr in enumerate(v)]
            dataframes_seq.append(pd.DataFrame(zip(data_lst), columns=[k]))


        dataframes_expand = []
        for k, v in expand_features.items():
            for i, usr in enumerate(v):
                exp_num = usr.shape[1] if len(usr.shape) == 2 else usr.shape[0]
                df_trx = pd.DataFrame([usr], columns=[f'{k}_{j:04d}' for j in range(exp_num)])
                dataframes_expand.append(df_trx)

        return dataframes_scalar, dataframes_seq, dataframes_expand

    @staticmethod
    def to_pandas_sequence(x, expand_features, scalar_features, seq_features, len_mask):
        dataframes_scalar = []
        for k, v in scalar_features.items():
            data_lst = [[data]*np.sum(len_mask[i]) for i, data in enumerate(v)]
            data_lst = list(chain(*data_lst))
            dataframes_scalar.append(pd.DataFrame(data_lst, columns=[k]))
        dataframes_scalar = pd.concat(dataframes_scalar, axis = 1)

        dataframes_seq = []
        for k, v in seq_features.items():
            data_lst = [data[len_mask[i]] for i, data in enumerate(v)]
            data_lst = list(chain(*data_lst))
            dataframes_seq.append(pd.DataFrame(data_lst, columns=[k]))

        dataframes_expand = []
        for k, v in expand_features.items():
            for i, usr in enumerate(v):
                exp_num = usr.shape[1] if len(usr.shape) == 2 else usr.shape[0]
                df_trx = pd.DataFrame(usr[len_mask[i]], columns=[f'{k}_{j:04d}' for j in range(exp_num)])
                dataframes_expand.append(df_trx)

        return dataframes_scalar, dataframes_seq, dataframes_expand


class InferenceModuleMultimodal(pl.LightningModule):
    def __init__(self, model, pandas_output=True, drop_seq_features=True, model_out_name='out', col_id = 'epk_id'):
        super().__init__()

        self.model = model
        self.pandas_output = pandas_output
        self.drop_seq_features = drop_seq_features
        self.model_out_name = model_out_name
        self.col_id = col_id

    def forward(self, x: PaddedBatch):
        x, batch_ids = x
        out = self.model(x)
        x_out = {self.col_id : batch_ids, self.model_out_name: out}
        if self.pandas_output:
            return self.to_pandas(x_out)
        return x_out

    @staticmethod
    def to_pandas(x):
        expand_cols = []
        scalar_features = {}

        for k, v in x.items():
            if type(v) is torch.Tensor:
                v = v.cpu().numpy()

            if type(v) is list or len(v.shape) == 1:
                scalar_features[k] = v
            elif len(v.shape) == 2:
                expand_cols.append(k)
            else:
                scalar_features[k] = None

        dataframes = [pd.DataFrame(scalar_features)]
        for col in expand_cols:
            v = x[col].cpu().numpy()
            dataframes.append(pd.DataFrame(v, columns=[f'{col}_{i:04d}' for i in range(v.shape[1])]))

        return pd.concat(dataframes, axis=1)
