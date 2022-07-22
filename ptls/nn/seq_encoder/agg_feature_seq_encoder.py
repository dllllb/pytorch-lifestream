from collections import OrderedDict

import torch

from ptls.data_load.padded_batch import PaddedBatch


class AggFeatureSeqEncoder(torch.nn.Module):
    """Calculates statistics over feature arrays and return them as embedding.

    Result is high dimension non learnable vector.
    Such embedding can be used in busting ML algorithm.

    Statistics are calculated by numerical features, with grouping by category values.

    This seq-encoder haven't TrxEncoder, they consume raw features.

    Parameters
        embeddings:
            dict with categorical feature names.
            Values must be like this `{'in': dictionary_size}`.
            One hot encoding will be applied to these features.
            Output vector size is (dictionary_size,)
        numeric_values:
            dict with numerical feature names.
            Values may contains some options
            numeric_values are multiplied with OHE categories.
            Then statistics are calculated over time dimensions.
        was_logified (bool):
            True - means that original numerical features was log transformed
            AggFeatureSeqEncoder will use `exp` to get original value
        log_scale_factor:
            Use it with `was_logified=True`. value will be multiplied by log_scale_factor before exp.
        is_used_count (bool):
            Use of not count by values
        is_used_mean (bool):
            Use of not mean by values
        is_used_std (bool):
            Use of not std by values
        use_topk_cnt (int):
            Define the K for topk features calculation. 0 if not used
        distribution_target_task (bool):
            Calc more features
        logify_sum_mean_seqlens (bool):
            True - apply log transform to sequence length

    Example:

    """
    def __init__(self,
                 embeddings=None,
                 numeric_values=None,
                 was_logified=True,
                 log_scale_factor=1,
                 is_used_count=True,
                 is_used_mean=True,
                 is_used_std=True,
                 use_topk_cnt=0,
                 distribution_target_task=False,
                 logify_sum_mean_seqlens=False,
                 ):

        super().__init__()

        self.numeric_values = OrderedDict(numeric_values.items())
        self.embeddings = OrderedDict(embeddings.items())
        self.was_logified = was_logified
        self.log_scale_factor = log_scale_factor
        self.distribution_target_task = distribution_target_task
        self.logify_sum_mean_seqlens = logify_sum_mean_seqlens

        self.eps = 1e-9

        self.ohe_buffer = {}
        for col_embed, options_embed in self.embeddings.items():
            size = options_embed['in']
            ohe = torch.diag(torch.ones(size))
            self.ohe_buffer[col_embed] = ohe
            self.register_buffer(f'ohe_{col_embed}', ohe)

        self.is_used_count = is_used_count
        self.is_used_mean = is_used_mean
        self.is_used_std = is_used_std
        self.use_topk_cnt = use_topk_cnt

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
        device = x.device
        B, T = x.seq_feature_shape
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

            if self.distribution_target_task:
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

            if not self.distribution_target_task:
                processed.append(std_)

            # TODO: percentiles

            # embeddings features (like mcc)
            for col_embed, options_embed in self.embeddings.items():
                ohe = getattr(self, f'ohe_{col_embed}')
                val_embed = feature_arrays[col_embed].long()
                val_embed = val_embed.clamp(0, options_embed['in'] - 1)

                ohe_transform = ohe[val_embed.flatten()].view(*val_embed.size(), -1)  # B, T, size
                m_sum = ohe_transform * val_orig.unsqueeze(-1)  # B, T, size

                # counts over val_embed
                mask = (1.0 - ohe[0]).unsqueeze(0)  # 0, 1, 1, 1, ..., 1

                e_cnt = ohe_transform.sum(dim=1) * mask
                if self.is_used_count:
                    processed.append(e_cnt)

                # sum over val_embed
                if self.is_used_mean:
                    e_sum = m_sum.sum(dim=1)
                    e_mean = e_sum.div(e_cnt + 1e-9)
                    processed.append(e_mean)

                if self.is_used_std:
                    a = torch.clamp(m_sum.pow(2).sum(dim=1) - m_sum.sum(dim=1).pow(2).div(e_cnt + 1e-9), min=0.0)
                    e_std = a.div(torch.clamp(e_cnt - 1, min=0) + 1e-9).pow(0.5)
                    processed.append(e_std)

        # n_unique and top_k
        for col_embed, options_embed in self.embeddings.items():
            ohe = getattr(self, f'ohe_{col_embed}')
            val_embed = feature_arrays[col_embed].long()
            val_embed = val_embed.clamp(0, options_embed['in'] - 1)

            ohe_transform = ohe[val_embed.flatten()].view(*val_embed.size(), -1)  # B, T, size

            # counts over val_embed
            mask = (1.0 - ohe[0]).unsqueeze(0)  # 0, 1, 1, 1, ..., 1
            e_cnt = ohe_transform.sum(dim=1) * mask

            processed.append(e_cnt.gt(0.0).float().sum(dim=1, keepdim=True))

            if self.use_topk_cnt > 0:
                cat_processed.append(torch.topk(e_cnt, self.use_topk_cnt, dim=1)[1])

        for i, t in enumerate(processed):
            if torch.isnan(t).any():
                raise Exception(f'nan in {i}')

        out = torch.cat(processed + cat_processed, 1)

        return out

    @property
    def embedding_size(self):
        numeric_values = self.numeric_values
        embeddings = self.embeddings

        e_sizes = [options_embed['in'] for col_embed, options_embed in embeddings.items()]

        out_size = 1

        n_features = sum([int(v) for v in [self.is_used_count, self.is_used_mean, self.is_used_std]])
        out_size += len(numeric_values) * (3 + n_features * sum(e_sizes)) + len(embeddings)
        out_size += len(e_sizes) * self.use_topk_cnt
        return out_size

    @property
    def cat_output_size(self):
        e_sizes = [options_embed['in'] for col_embed, options_embed in self.embeddings.items()]
        return len(e_sizes) * self.use_topk_cnt

    @property
    def category_names(self):
        return set([field_name for field_name in self.embeddings.keys()] +
                   [value_name for value_name in self.numeric_values.keys()]
                   )

    @property
    def category_max_size(self):
        return {k: v['in'] for k, v in self.embeddings.items()}
