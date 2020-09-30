from collections import OrderedDict

import torch

from dltranz.trx_encoder import PaddedBatch


class AggFeatureModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.numeric_values = OrderedDict(config['numeric_values'].items())
        self.embeddings = OrderedDict(config['embeddings'].items())
        self.was_logified = config['was_logified']
        self.log_scale_factor = config['log_scale_factor']

        self.eps = 1e-9

        self.ohe_buffer = {}
        for col_embed, options_embed in self.embeddings.items():
            size = options_embed['in']
            ohe = torch.diag(torch.ones(size))
            self.ohe_buffer[col_embed] = ohe
            self.register_buffer(f'ohe_{col_embed}', ohe)

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

        processed = [seq_lens.unsqueeze(1)]  # count

        for col_num, options_num in self.numeric_values.items():
            # take array with numerical feature and convert it to original scale
            if col_num == '#ones':
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
            processed.append(val_orig.sum(dim=1).unsqueeze(1))  # sum
            processed.append(val_orig.sum(dim=1).div(seq_lens + self.eps).unsqueeze(1))  # mean
            a = torch.clamp(val_orig.pow(2).sum(dim=1) - val_orig.sum(dim=1).pow(2).div(seq_lens + self.eps), min=0.0)
            processed.append(a.div(torch.clamp(seq_lens - 1, min=0.0) + self.eps).pow(0.5).unsqueeze(1))  # std

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
                processed.append(e_cnt)

                # sum over val_embed
                e_sum = m_sum.sum(dim=1)
                e_mean = e_sum.div(e_cnt + 1e-9)
                processed.append(e_mean)

                a = torch.clamp(m_sum.pow(2).sum(dim=1) - m_sum.sum(dim=1).pow(2).div(e_cnt + 1e-9), min=0.0)
                e_std = a.div(torch.clamp(e_cnt - 1, min=0) + 1e-9).pow(0.5)
                processed.append(e_std)

        # n_unique
        for col_embed, options_embed in self.embeddings.items():
            ohe = getattr(self, f'ohe_{col_embed}')
            val_embed = feature_arrays[col_embed].long()

            ohe_transform = ohe[val_embed.flatten()].view(*val_embed.size(), -1)  # B, T, size

            # counts over val_embed
            mask = (1.0 - ohe[0]).unsqueeze(0)  # 0, 1, 1, 1, ..., 1
            e_cnt = ohe_transform.sum(dim=1) * mask

            processed.append(e_cnt.gt(0.0).float().sum(dim=1, keepdim=True))

        for i, t in enumerate(processed):
            if torch.isnan(t).any():
                raise Exception(f'nan in {i}')

        out = torch.cat(processed, 1)
        return out

    @staticmethod
    def output_size(config):
        numeric_values = config['numeric_values']
        embeddings = config['embeddings']

        e_sizes = [options_embed['in'] for col_embed, options_embed in embeddings.items()]

        return 1 + len(numeric_values) * (3 + 3 * sum(e_sizes)) + len(embeddings)
