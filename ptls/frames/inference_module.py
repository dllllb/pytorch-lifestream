import os
import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np
import onnxruntime as ort

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
    
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def to_pandas(self, x):
        if isinstance(x, PaddedBatch):
            len_mask = x.seq_len_mask.bool().cpu().numpy()
            payload = x.payload
        else:
            payload = x
            len_mask = None

        is_reduced = not isinstance(payload[self.model_out_name], PaddedBatch)

        scalar_feats, seq_feats, expand_feats = {}, {}, {}
        for k, v in payload.items():
            if isinstance(v, PaddedBatch):
                len_mask = v.seq_len_mask.bool().cpu().numpy()
                arr = v.payload
            else:
                arr = v
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()

            if isinstance(arr, list) or arr.ndim == 1:
                scalar_feats[k] = np.asarray(arr)
            elif arr.ndim == 3 or (k == self.model_out_name and arr.ndim == 2):
                expand_feats[k] = arr
            else:
                seq_feats[k] = arr

        if is_reduced:
            return self._record_to_pandas(scalar_feats, seq_feats, expand_feats, len_mask)
        else:
            return self._sequence_to_pandas(scalar_feats, seq_feats, expand_feats, len_mask)

    @staticmethod
    def _record_to_pandas(scalar_feats, seq_feats, expand_feats, len_mask):
        df = pd.DataFrame(scalar_feats)
        # Add sequence features
        for k, arr in seq_feats.items():
            df[k] = [arr[i][:np.sum(len_mask[i])] for i in range(arr.shape[0])]
        # Add expanded features
        for k, arr in expand_feats.items():
            # (batch_size, features) -> (batch_size, -1)
            flat = arr.reshape(arr.shape[0], -1)
            cols = [f'{k}_{j:04d}' for j in range(flat.shape[1])]
            df_expand = pd.DataFrame(flat, columns=cols, index=df.index)
            df = pd.concat([df, df_expand], axis=1)
        return df

    @staticmethod
    def _sequence_to_pandas(scalar_feats, seq_feats, expand_feats, len_mask):
        lengths = len_mask.sum(axis=1).astype(int)
        total = int(lengths.sum())
        data = {}

        # Repeat scalar features per sequence length
        for k, arr in scalar_feats.items():
            data[k] = np.repeat(arr, lengths)

        # Flatten sequence features
        mask_flat = len_mask.flatten()
        for k, arr in seq_feats.items():
            data[k] = arr.flatten()[mask_flat]

        # Flatten expanded features
        for k, arr in expand_feats.items():
            samples = []
            for i in range(arr.shape[0]):
                samples.append(arr[i, :lengths[i]].reshape(lengths[i], -1))
            flat = np.vstack(samples)
            for j in range(flat.shape[1]):
                data[f'{k}_{j:04d}'] = flat[:, j]

        return pd.DataFrame(data)


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

class ONNXInferenceModule(InferenceModule):
    def __init__(self, model, dl, model_out_name='emb.onnx', pandas_output=False):
        super().__init__(model)
        self.model = model
        self.pandas_output = pandas_output
        self.model_out_name = model_out_name
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        batch = next(iter(dl))
        features, names, seq_len = self.preprocessing(batch)
        model._col_names = names
        model._seq_len = seq_len
        model.example_input_array = features
        self.export(self.model_out_name, model)
    
        self.ort_session = ort.InferenceSession(
            self.model_out_name,
            providers=self.providers
        )
    
    def stack(self, x):
        x = [v for v in x[0].values()]
        return torch.stack(x)
    
    def preprocessing(self, x):
        features = self.stack(x)
        names = [k for k in x[0].keys()]
        seq_len = x[1]
        return features, names, seq_len

    def export(self,
                path: str,
                model
            ) -> None:
        
        model.to_onnx(path,
                    export_params=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                                "input": {
                                    0: "features",
                                    1: "batch_size",
                                    2: "seq_len"
                                },
                                "output": {
                                    0: "batch_size",
                                    1: "hidden_size"
                                }
                            }
                    )
    
    def forward(self, x, dtype: torch.dtype = torch.float16):
        inputs = self.to_numpy(self.stack(x))
        out = self.ort_session.run(None, {"input": inputs})
        out = torch.tensor(out[0], dtype=dtype)
        if self.pandas_output:
            return self.to_pandas(out)
        return out

    def to(self, device):
        return self

    def size(self):
        return os.path.getsize(self.model_name)
    
    def predict(self, dl, dtype: torch.dtype = torch.float16):
        pred = list()
        with torch.no_grad():
            for batch in dl:
                output = self(batch, dtype=dtype)
                pred.append(output)
        return pred