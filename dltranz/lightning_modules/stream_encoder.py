import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm


class TBatchNorm(torch.nn.BatchNorm1d):
    def forward(self, x):
        B, T, C = x.size()
        return super().forward(x.view(B * T, C)).view(B, T, C)


class TDropout(torch.nn.Module):
    def __init__(self, p, one_for_batch=False):
        super().__init__()
        self.p = p
        self.one_for_batch = one_for_batch

    def forward(self, x):
        if not self.training:
            return x
        if self.p > 0:
            B = 1 if self.one_for_batch else x.size(0)
            x_mask = torch.bernoulli(torch.ones(B, 1, x.size(2), device=x.device) * self.p)
            x_mask /= 1 - self.p
            x = x * x_mask
        return x


class ClipRange(torch.nn.Module):
    def __init__(self, clip_range):
        super().__init__()
        self.clip_range = clip_range

    def forward(self, x):
        x = torch.clamp(x, *self.clip_range)
        return x


class DummyLayer(torch.nn.Module):
    def forward(self, x):
        return x


class StreamEncoder(pl.LightningModule):
    def __init__(self, encoder_x2z,
                 history_size, predict_range,
                 z_channels, c_channels,
                 var_gamma_z, var_gamma_c,
                 lr, weight_decay, step_size, gamma,
                 cpc_w, cov_z_w, var_z_w, cov_c_w, var_c_w,
                 ):
        super().__init__()

        self.save_hyperparameters()  # ignore='encoder_x2z'

        self.encoder_x2z = encoder_x2z

        self.ar_rnn_z2c = torch.nn.RNN(
            input_size=z_channels,
            hidden_size=c_channels,
            batch_first=True,
        )

        self.lin_predictors_c2p = torch.nn.ModuleList([
            torch.nn.Linear(c_channels, z_channels) for _ in predict_range
        ])

        self.reg_bn_z = torch.nn.BatchNorm1d(z_channels, affine=False)
        self.reg_bn_c = torch.nn.BatchNorm1d(c_channels, affine=False)

    def configure_optimizers(self):
        optimisers = [torch.optim.Adam(self.parameters(),
                                       lr=self.hparams.lr,
                                       weight_decay=self.hparams.weight_decay,
                                       )]
        schedulers = [torch.optim.lr_scheduler.StepLR(o,
                                                      step_size=self.hparams.step_size,
                                                      gamma=self.hparams.gamma) for o in optimisers]
        return optimisers, schedulers

    def forward(self, x):
        z = self.encoder_x2z(x)
        c, h = self.ar_rnn_z2c(z)
        return z, c

    def training_step(self, batch, batch_idx):
        x = batch[0]
        z = self.encoder_x2z(x)

        zx, zy = z[:, :self.hparams.history_size], z[:, self.hparams.history_size:]

        cpc_loss, cx = self.cpc_loss(zx, zy)
        cov_z_loss = self.cov_z_loss(zx)
        var_z_loss = self.var_z_loss(zx)
        cov_c_loss = self.cov_c_loss(cx)
        var_c_loss = self.var_c_loss(cx)
        loss = 0.0
        if self.hparams.cpc_w > 0:
            loss += self.hparams.cpc_w * cpc_loss
        if self.hparams.cov_c_w > 0:
            loss += self.hparams.cov_c_w * cov_c_loss
        if self.hparams.var_c_w > 0:
            loss += self.hparams.var_c_w * var_c_loss
        if self.hparams.cov_z_w > 0:
            loss += self.hparams.cov_z_w * cov_z_loss
        if self.hparams.var_z_w > 0:
            loss += self.hparams.var_z_w * var_z_loss

        self.log('loss/cpc', cpc_loss, prog_bar=True)
        self.log('loss/c_cov', cov_c_loss, prog_bar=True)
        self.log('loss/c_var', var_c_loss, prog_bar=True)
        self.log('loss/z_cov', cov_z_loss, prog_bar=True)
        self.log('loss/z_var', var_z_loss, prog_bar=True)
        self.log('loss/loss', loss)

        return loss

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {
            'hp/z_std': 0,
            'hp/cpc_dtr': 0,
            'hp/cpc_dtt': 0,
            'hp/cpc_d_lift': 0,
            'hp/cpc_r2': 0,
            'hp/cpc_pow': 0,
            # "hp/zf_cor": 0,
            # "hp/ze_cor": 0,
            # "hp/cf_cor": 0,
            # "hp/ce_cor": 0,
            'hp/z_self_corr': 0,
            'hp/z_unique_features': 0,
            'hp/c_self_corr': 0,
            'hp/c_unique_features': 0,
        })

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        z, c = self.forward(x)

        z_std = z.std(dim=1).mean()

        t = torch.randn(10000, 1, device=z.device).repeat(1, z.size(2)) * z_std
        t = t[1:] - t[:-1]
        dtr = t.pow(2).sum(dim=1).pow(0.5).mean()

        history_size = self.hparams.history_size
        dtt = 0.0
        r2_score = 0
        for i, l in zip(self.hparams.predict_range, self.lin_predictors_c2p):
            t = z[:, history_size + i:, :]

            p = l(c[:, history_size - 1 : z.size(1) - (i + 1), :])
            dtt += (p - t).pow(2).sum(dim=2).pow(0.5).mean()

            r2_s = torch.where(
                t.std(dim=1) > 1e-3,
                1 - (p - t).pow(2).mean(dim=1) / t.var(dim=1),
                torch.tensor([0.0], device=x.device),
            )
            r2_score += r2_s.mean()
        dtt /= len(self.lin_predictors_c2p)
        cpc_r2 = r2_score / len(self.lin_predictors_c2p)

        self.log('hp/z_std', z_std)
        self.log('hp/cpc_dtr', dtr)
        self.log('hp/cpc_dtt', dtt)
        self.log('hp/cpc_d_lift', (dtr - dtt) / (dtr + 1e-6))
        self.log('hp/cpc_r2', cpc_r2)

        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        z = (z - z.mean(dim=1, keepdim=True)) / (z.std(dim=1, keepdim=True) + 1e-6)
        c = (c - c.mean(dim=1, keepdim=True)) / (c.std(dim=1, keepdim=True) + 1e-6)

        # m = (torch.bmm(x.transpose(1, 2), z) / x.size(1)).abs()  # B, x, z
        # self.log('hp/zf_cor', m.max(dim=2).values.mean())
        # self.log('hp/ze_cor', m.max(dim=1).values.mean())
        #
        # m = (torch.bmm(x.transpose(1, 2), c) / x.size(1)).abs()  # B, x, c
        # self.log('hp/cf_cor', m.max(dim=2).values.mean())
        # self.log('hp/ce_cor', m.max(dim=1).values.mean())

        mz = (torch.bmm(z.transpose(1, 2), z) / z.size(1)).abs()  # B, z, z
        Cz = mz.size(1)
        off_diag_ix = (1 - torch.eye(Cz, device=x.device)).bool().view(-1)
        m = mz.view(-1, Cz * Cz)[:, off_diag_ix]
        self.log('hp/z_self_corr', 0.0 if Cz == 1 else m.mean())
        self.log('hp/z_unique_features', 1 / (mz.mean() + 1e-3))
        self.log('hp/cpc_pow', cpc_r2 / (mz.mean() + 1e-3))

        mc = (torch.bmm(c.transpose(1, 2), c) / c.size(1)).abs()  # B, z, z
        Cc = mc.size(1)
        off_diag_ix = (1 - torch.eye(Cc, device=x.device)).bool().view(-1)
        m = mc.view(-1, Cc * Cc)[:, off_diag_ix]
        self.log('hp/c_self_corr', 0.0 if Cz == 1 else m.mean())
        self.log('hp/c_unique_features', 1 / (mc.mean() + 1e-3))

    def cpc_loss(self, x, y):
        out, h = self.ar_rnn_z2c(x)
        c = out[:, -1, :]  # B, Hc

        loss = 0.0
        for i, l in zip(self.hparams.predict_range, self.lin_predictors_c2p):
            p = l(c)
            t = y[:, i]
            loss += (p - t).pow(2).sum(dim=1).mean()

        return loss / len(self.lin_predictors_c2p), c

    def cov_z_loss(self, x):
        B, T, C = x.size()
        if C == 1:
            return torch.zeros(1, dtype=x.dtype, device=x.device).mean()
        x = self.reg_bn_z(x.reshape(B * T, C)).reshape(B, T, C)
        m = torch.bmm(x.transpose(1, 2), x) / T  # B, C, C
        off_diag_ix = (1 - torch.eye(C, device=x.device)).bool().view(-1)
        loss = m.view(B, C * C)[:, off_diag_ix].pow(2).mean()
        return loss

    def var_z_loss(self, x):
        B, T, C = x.size()
        v = (torch.var(x, dim=1) + 1e-6).pow(0.5)
        loss = torch.relu(self.hparams.var_gamma_z - v).mean()
        return loss

    def cov_c_loss(self, x):
        B, C = x.size()
        if C == 1:
            return torch.zeros(1, dtype=x.dtype, device=x.device).mean()
        x = self.reg_bn_c(x)
        m = torch.mm(x.T, x) / B  # C, C
        off_diag_ix = (1 - torch.eye(C, device=x.device)).bool().view(-1)
        loss = m.view(C * C)[off_diag_ix].pow(2).mean()
        return loss

    def var_c_loss(self, x):
        v = (torch.var(x, dim=0) + 1e-6).pow(0.5)
        loss = torch.relu(self.hparams.var_gamma_c - v).mean()
        return loss


class Loader3DTensor:
    """Expected 3D tensor Batch x Time x Channels
    Data should be normalized along Time axis for each individual sample in batch
    """
    def __init__(self, stream_encoder):
        self.history_size = stream_encoder.hparams.history_size
        self.predict_size = max(stream_encoder.hparams.predict_range) + 1

    def get_train_dataloader(self, data, batch_size, num_workers):
        def gen_batches(data):
            B, T, C = data.size()
            sample_len = self.history_size + self.predict_size
            for b in tqdm(range(B)):
                for i in tqdm(range(0, T - sample_len)):
                    yield data[b, i:i + sample_len]

        all_batches = torch.stack(list(gen_batches(data)))

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(all_batches),
            shuffle=True,
            batch_size=batch_size,
            persistent_workers=True,
            num_workers=num_workers,
        )

    @staticmethod
    def get_valid_dataloader(data, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data),
            batch_size=batch_size,
            num_workers=num_workers,
        )


class LoaderMultiIndexPandas:
    """
    Expected pandas dataframe vhere values is a features, and index is (region, date)
    date is sorted and regular (with missing dates restored)
    Data should be normalized along Time axis for each individual region
    """
    def __init__(self, stream_encoder):
        self.history_size = stream_encoder.hparams.history_size
        self.predict_size = max(stream_encoder.hparams.predict_range) + 1

    def get_train_dataloader(self, data, batch_size, num_workers):
        def gen_batches(data):
            index_name = data.index.names[0]
            sample_len = self.history_size + self.predict_size
            for batch_ix in tqdm(data.reset_index(index_name)[index_name].drop_duplicates().values):
                df = data.loc[batch_ix].values
                T, C = df.shape
                for i in range(0, T - sample_len):
                    yield torch.from_numpy(df[i:i + sample_len].astype(np.float32))

        all_batches = torch.stack(list(gen_batches(data)))

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(all_batches),
            shuffle=True,
            batch_size=batch_size,
            persistent_workers=False,
            num_workers=num_workers,
        )

    @staticmethod
    def get_valid_dataloader(data, batch_size, num_workers):
        def gen_batches(data):
            index_name = data.index.names[0]
            for batch_ix in tqdm(data.reset_index(index_name)[index_name].drop_duplicates().values):
                df = data.loc[batch_ix].values
                yield torch.from_numpy(df.astype(np.float32))

        all_batches = list(gen_batches(data))
        all_lenghts = [len(t) for t in all_batches]
        all_batches = torch.nn.utils.rnn.pad_sequence(all_batches, batch_first=True)

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(all_batches),
            batch_size=batch_size,
            num_workers=num_workers,
        ), all_lenghts
