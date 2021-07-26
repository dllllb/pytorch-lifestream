import pickle
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class TBatchNorm(torch.nn.BatchNorm1d):
    def forward(self, x):
        B, T, C = x.size()
        return super().forward(x.view(B * T, C)).view(B, T, C)


class InputNorm(torch.nn.Module):
    def __init__(self, num_channels, clip_range):
        super().__init__()
        self.bn = TBatchNorm(num_channels)
        self.clip_range = clip_range

    def forward(self, x):
        B, T, C = x.size()
        x = self.bn(x)
        x = torch.clamp(x, *self.clip_range)
        return x


class StreamEncoder(pl.LightningModule):
    def __init__(self,
                 train_batch_size, history_size, predict_size,
                 in_channels, clip_range,
                 z_channels,
                 c_channels,
                 var_gamma,
                 lr, weight_decay, step_size, gamma,
                 cpc_w, cov_w, var_w,
                 plot_samples=[],
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.input_model = InputNorm(in_channels, clip_range)

        self.cnn_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, z_channels),
            TBatchNorm(z_channels),
        )

        self.ar_rnn = torch.nn.GRU(
            input_size=z_channels,
            hidden_size=c_channels,
            batch_first=True,
        )

        self.lin_predictors = torch.nn.ModuleList([
            torch.nn.Linear(c_channels, z_channels) for _ in range(predict_size)
        ])

        self.reg_bn = torch.nn.BatchNorm1d(c_channels, affine=False)

    def get_train_dataloader(self, data, batch_size, num_workers):
        def gen_batches(data):
            B, T, C = data.size()
            sample_len = self.hparams.history_size + self.hparams.predict_size
            for b in range(B):
                for i in range(0, T - sample_len):
                    yield data[b, i:i + sample_len]

        all_batches = torch.stack(list(gen_batches(data)))

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(all_batches),
            shuffle=True,
            batch_size=batch_size,
            persistent_workers=True,
            num_workers=num_workers,
        )

    def get_valid_dataloader(self, data, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data),
            batch_size=batch_size,
            num_workers=num_workers,
        )

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
        x = self.input_model(x)
        z = self.cnn_encoder(x)
        c, h = self.ar_rnn(z)
        return x, z, c

    def training_step(self, batch, batch_idx):
        x = self.input_model(batch[0])
        z = self.cnn_encoder(x)

        zx, zy = z[:, :self.hparams.history_size], z[:, self.hparams.history_size:]

        cpc_loss, cx = self.cpc_loss(zx, zy)
        cov_loss = self.cov_loss(cx)
        var_loss = self.var_loss(cx)
        loss = 0.0
        if self.hparams.cpc_w > 0:
            loss += self.hparams.cpc_w * cpc_loss
        if self.hparams.cov_w > 0:
            loss += self.hparams.cov_w * cov_loss
        if self.hparams.var_w > 0:
            loss += self.hparams.var_w * var_loss

        self.log('cpc_loss', cpc_loss, prog_bar=True)
        self.log('cov_loss', cov_loss, prog_bar=True)
        self.log('var_loss', var_loss, prog_bar=True)
        self.log('loss', loss)

        return loss

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {
            "hp/zf_cor": 0,
            "hp/ze_cor": 0,
            "hp/cf_cor": 0,
            "hp/ce_cor": 0,
        })

    def plot_xzc(self, x, z, c):
        x = x[0, :500].detach().cpu().numpy()
        z = z[0, :500].detach().cpu().numpy()
        c = c[0, :500].detach().cpu().numpy()

        # plot signal
        fig, axs = plt.subplots(3, 1, figsize=(12, 4 * 3))
        axs[0].plot(x)
        axs[0].set_title('x signal')
        axs[1].plot(z)
        axs[1].set_title('z signal')
        axs[2].plot(c)
        axs[2].set_title('c signal')
        plt.suptitle('embedding signals')
        self.logger.experiment.add_figure('Signals', fig, global_step=self.global_step)

        # plot spectrum
        fig, axs = plt.subplots(3, 1, figsize=(12, 3 * 12 * x.shape[1] / x[0, :500].shape[0]))
        axs[0].imshow(x.T)
        axs[0].set_title('x spectrum')
        axs[1].imshow(z.T)
        axs[1].set_title('z spectrum')
        axs[2].imshow(c.T)
        axs[2].set_title('c spectrum')
        self.logger.experiment.add_figure('Spectrum', fig, global_step=self.global_step)

    def validation_step(self, batch, batch_idx):
        x, z, c = self.forward(batch[0])

        z = z - z.mean(dim=1, keepdim=True)
        z = z / z.std(dim=1, keepdim=True)
        c = c - c.mean(dim=1, keepdim=True)
        c = c / c.std(dim=1, keepdim=True)

        m = (torch.bmm(x.transpose(1, 2), z) / x.size(1)).abs()
        self.log('hp/zf_cor', m.max(dim=2).values.mean())
        self.log('hp/ze_cor', m.max(dim=1).values.mean())

        m = (torch.bmm(x.transpose(1, 2), c) / x.size(1)).abs()
        self.log('hp/cf_cor', m.max(dim=2).values.mean())
        self.log('hp/ce_cor', m.max(dim=1).values.mean())

    def cpc_loss(self, x, y):
        out, h = self.ar_rnn(x)
        c = h[0]  # B, Hc

        loss = 0.0
        for i, l in enumerate(self.lin_predictors):
            p = l(c)
            t = y[:, i]
            loss += (p - t).pow(2).sum(dim=1).mean()

        return loss / self.hparams.predict_size, c

    def cov_loss(self, x):
        B, C = x.size()
        x = self.reg_bn(x)
        m = torch.mm(x.T, x) / B  # C, C
        off_diag_ix = (1 - torch.eye(C, device=x.device)).bool().view(-1)
        loss = m.view(C * C)[off_diag_ix].pow(2).mean()
        return loss

    def var_loss(self, x):
        v = (torch.var(x, dim=0) + 1e-6).pow(0.5)
        loss = torch.relu(self.hparams.var_gamma - v).mean()
        return loss
