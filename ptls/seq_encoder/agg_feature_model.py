from collections import OrderedDict

import torch
import numpy as np

from ptls.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.trx_encoder import PaddedBatch


class AggFeatureModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.numeric_values = OrderedDict(config['numeric_values'].items())
        self.embeddings = OrderedDict(config['embeddings'].items())
        self.was_logified = config['was_logified']
        self.log_scale_factor = config['log_scale_factor']
        self.distribution_targets_task = config.get('distribution_targets_task', False)
        self.logify_sum_mean_seqlens = config.get('logify_sum_mean_seqlens', False)

        self.eps = 1e-9

        self.ohe_buffer = {}
        for col_embed, options_embed in self.embeddings.items():
            size = options_embed['in']
            ohe = torch.diag(torch.ones(size))
            self.ohe_buffer[col_embed] = ohe
            self.register_buffer(f'ohe_{col_embed}', ohe)

        self.num_aggregators = config.get('num_aggregators', {'count': True, 'sum': True, 'std': True})
        self.cat_aggregators = config.get('cat_aggregators', {})

    def forward(self, x: PaddedBatch):
        """
        {
            'cat_i': [B, T]: int
            'num_i': [B, T]: float
        }
        to
        {
            [B, H] where H - is [f1, f2, f3, ... fn]
        }
        :param x:
        :return:
        """

        feature_arrays = x.payload
        device = next(iter(feature_arrays.values())).device
        B, T = next(iter(feature_arrays.values())).size()
        seq_lens = x.seq_lens.to(device).float()
        # if (seq_lens == 0).any():
        #     raise Exception('seq_lens == 0')

        if (self.logify_sum_mean_seqlens):
            processed = [torch.log(seq_lens.unsqueeze(1))]  # count
        else:
            processed = [seq_lens.unsqueeze(1)]
        cat_processed = []

        for col_num, options_num in self.numeric_values.items():
            # take array with numerical feature and convert it to original scale
            if col_num.strip('"') == '#ones':
                val_orig = torch.ones(B, T, device=device)
            else:
                val_orig = feature_arrays[col_num].float()

            if any((
                    type(options_num) is str and self.was_logified,
                    type(options_num) is dict and options_num.get('was_logified', False),
            )):
                #
                val_orig = torch.expm1(self.log_scale_factor * torch.abs(val_orig)) * torch.sign(val_orig)

            # trans_common_features

            sum_ = val_orig.sum(dim=1).unsqueeze(1)  # sum
            a = torch.clamp(val_orig.pow(2).sum(dim=1) - val_orig.sum(dim=1).pow(2).div(seq_lens + self.eps), min=0.0)
            mean_ = val_orig.sum(dim=1).div(seq_lens + self.eps).unsqueeze(1)  # mean
            std_ = a.div(torch.clamp(seq_lens - 1, min=0.0) + self.eps).pow(0.5).unsqueeze(1)  # std
   
            if self.distribution_targets_task:
                sum_pos = torch.clamp(val_orig, min=0).sum(dim=1).unsqueeze(1)
                processed.append(torch.log(sum_pos + 1))  # log sum positive

                sum_neg = torch.clamp(val_orig, max=0).sum(dim=1).unsqueeze(1)
                processed.append(-1 * torch.log(-sum_neg + 1))  # log sum negative
            elif self.logify_sum_mean_seqlens:
                processed.append(torch.sign(sum_) * torch.log(torch.abs(sum_) + 1))
            else:
                processed.append(sum_)

            if self.logify_sum_mean_seqlens:
                mean_ = torch.sign(mean_) * torch.log(torch.abs(mean_) + 1)

            processed.append(mean_)

            if not self.distribution_targets_task:
                processed.append(std_)

            # TODO: percentiles

            # embeddings features (like mcc)
            for col_embed, options_embed in self.embeddings.items():
                ohe = getattr(self, f'ohe_{col_embed}')
                val_embed = feature_arrays[col_embed].long()

                ohe_transform = ohe[val_embed.flatten()].view(*val_embed.size(), -1)  # B, T, size
                m_sum = ohe_transform * val_orig.unsqueeze(-1)  # B, T, size

                # counts over val_embed
                mask = (1.0 - ohe[0]).unsqueeze(0)  # 0, 1, 1, 1, ..., 1

                e_cnt = ohe_transform.sum(dim=1) * mask
                if self.num_aggregators.get('count', False):
                    processed.append(e_cnt)

                # sum over val_embed
                if self.num_aggregators.get('sum', False):
                    e_sum = m_sum.sum(dim=1)
                    e_mean = e_sum.div(e_cnt + 1e-9)
                    processed.append(e_mean)

                if self.num_aggregators.get('std', False):
                    a = torch.clamp(m_sum.pow(2).sum(dim=1) - m_sum.sum(dim=1).pow(2).div(e_cnt + 1e-9), min=0.0)
                    e_std = a.div(torch.clamp(e_cnt - 1, min=0) + 1e-9).pow(0.5)
                    processed.append(e_std)

        # n_unique and top_k
        for col_embed, options_embed in self.embeddings.items():
            ohe = getattr(self, f'ohe_{col_embed}')
            val_embed = feature_arrays[col_embed].long()

            ohe_transform = ohe[val_embed.flatten()].view(*val_embed.size(), -1)  # B, T, size

            # counts over val_embed
            mask = (1.0 - ohe[0]).unsqueeze(0)  # 0, 1, 1, 1, ..., 1
            e_cnt = ohe_transform.sum(dim=1) * mask

            processed.append(e_cnt.gt(0.0).float().sum(dim=1, keepdim=True))

            if self.cat_aggregators and self.cat_aggregators['top_k']:
                cat_processed.append(torch.topk(e_cnt, self.cat_aggregators['top_k'], dim=1)[1])

        for i, t in enumerate(processed):
            if torch.isnan(t).any():
                raise Exception(f'nan in {i}')
    
        out = torch.cat(processed + cat_processed, 1)

        return out

    @property
    def output_size(self):
        numeric_values = self.numeric_values
        embeddings = self.embeddings

        e_sizes = [options_embed['in'] for col_embed, options_embed in embeddings.items()]

        out_size = 1

        n_features = sum(self.num_aggregators.values())
        out_size += len(numeric_values) * (3 + n_features * sum(e_sizes)) + len(embeddings)
        out_size += len(e_sizes) * self.cat_aggregators.get('top_k', 0)
        return out_size

    @property
    def cat_output_size(self):
        e_sizes = [options_embed['in'] for col_embed, options_embed in self.embeddings.items()]
        return len(e_sizes) * self.cat_aggregators.get('top_k', 0)

    @property
    def category_names(self):
        return set([field_name for field_name in self.embeddings.keys()] +
                   [value_name for value_name in self.numeric_values.keys()]
                   )

    @property
    def category_max_size(self):
        return {k: v['in'] for k, v in self.embeddings.items()}


class AggFeatureSeqEncoder(AbsSeqEncoder):
    def __init__(self, params, is_reduce_sequence):
        super().__init__(params, is_reduce_sequence)

        self.model = AggFeatureModel(params['trx_encoder'])

    @property
    def is_reduce_sequence(self):
        return self._is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        if not value:
            raise NotImplementedError('Only sequence embedding can be returned')
        self._is_reduce_sequence = value

    @property
    def category_max_size(self):
        return self.model.category_max_size

    @property
    def category_names(self):
        return self.model.category_names

    @property
    def embedding_size(self):
        return self.model.output_size

    def forward(self, x):
        x = self.model(x)
        return x
