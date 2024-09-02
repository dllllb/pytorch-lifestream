from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import ClusterLoss
from ptls.frames.coles.metric import BatchAccuracy, BatchRecallTopK
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.data_load.padded_batch import PaddedBatch
import torch
from torch import nn
import numpy as np
from pytorch_lightning.callbacks import Callback
import contextlib
import sys
from collections import defaultdict
from functools import partial
import warnings

try:
    import faiss
except:
    warnings.warn("Warning: faiss module import failed. Clustering modules are unavailable.")


class DummyFile(object):
    def write(self, x):
        pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


class ClusterModule(ABSModule):
    def __init__(self,
                 seq_encoder: SeqEncoderContainer = None,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 num_cluster=None):
        if head is None:
            head = Head(use_norm_encoder=True)

        if loss is None:
            loss = ClusterLoss()

        if validation_metric is None:
            validation_metric = {"cluster": defaultdict(partial(BatchAccuracy, k=1)),
                                 "coles": BatchRecallTopK(K=4, metric='cosine')}

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        self._head = head
        self.train_clusters = {'idx2cluster': [], 'centroids': [], 'density': []}
        self.val_clusters = {'idx2cluster': [], 'centroids': [], 'density': []}

        assert type(num_cluster) == list
        self._num_cluster = num_cluster

    @property
    def metric_name(self):
        return 'topk_accuracy'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def shared_step(self, x, y):
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, y

    def training_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        loss, info = self._loss(y_h, y, self.train_clusters)
        for k, v in info.items():
            self.log(k, v)

        if type(batch) is tuple:
            x, y = batch
            if isinstance(x, PaddedBatch):
                self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True, logger=True)
        else:
            # this code should not be reached
            self.log('seq_len', -1, prog_bar=True)
            raise AssertionError('batch is not a tuple')
        return loss

    def log_metrics(self, info):
        for k, v in info.items():
            if k.startswith('CLUSTER'):
                size = k.split('_')[1]
                logs, labs = v
                self._validation_metric["cluster"][size](logs, labs)

    def validation_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        self._validation_metric['coles'](y_h, y['coles_target'])
        _, info = self._loss(y_h, y, self.val_clusters)
        self.log_metrics(info)

    def write_metrics(self, m, name=""):
        if type(m) in [dict, defaultdict]:
            for k, v in m.items():
                self.write_metrics(v, " ".join([name, str(k)]))
        else:
            self.log(name, m.compute(), logger=True)
            m.reset()

    def on_validation_epoch_end(self):
        self.write_metrics(self._validation_metric, "")


class ClusterCallback(Callback):
    def __init__(self, cluster_datamodule, num_cluster, temperature, device,
                 use_portion_to_train=None, run_each_n_train_steps=None):
        self.data = cluster_datamodule

        assert type(device) is int
        assert type(num_cluster) in [list, int]
        if type(num_cluster) == int:
            num_cluster = [num_cluster]
        self.num_cluster = num_cluster
        self.temperature = temperature
        self.device = device
        self.use_portion_to_train = use_portion_to_train if use_portion_to_train is not None else 1.
        self.run_each_n_train_steps = run_each_n_train_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.run_each_n_train_steps is not None:
            if batch_idx % self.run_each_n_train_steps == 0:
                self.train_clustering(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.run_each_n_train_steps is None:
            self.train_clustering(pl_module)

    def train_clustering(self, pl_module):
        with torch.no_grad():
            feats = self.compute_features(self.data.train_dataloader(), pl_module)
            train_clusters = self.run_kmeans(feats)
            for num_clus, prev, now in zip(self.num_cluster,
                                           pl_module.train_clusters['idx2cluster'],
                                           train_clusters['idx2cluster']):
                cluster_switch_portion = 1 - ((prev == now).sum() / prev.shape[0])
                pl_module.log("train_cluster_change_" + str(num_clus), cluster_switch_portion,
                              prog_bar=True, logger=True)
            pl_module.train_clusters = train_clusters

    def on_validation_epoch_start(self, trainer, pl_module):
        with torch.no_grad():
            feats = self.compute_features(self.data.val_dataloader(), pl_module)
            pl_module.val_clusters = self.run_kmeans(feats)

    def compute_features(self, loader, model):
        features, idx = list(), list()
        for batch in loader:
            feats, y = model.shared_step(batch[0].to(self.device), batch[1])
            features.extend(list(feats.cpu().numpy()))
            idx.extend(list(y["cluster_target"].cpu().numpy()))
        return np.stack(features, axis=0)[np.argsort(idx)]

    def run_kmeans(self, x):
        results = {'idx2cluster': [], 'centroids': [], 'density': []}

        if self.use_portion_to_train < 1:
            x_train = x[np.random.choice(x.shape[0], int(x.shape[0] * self.use_portion_to_train), replace=False)]
        else:
            x_train = x

        for seed, num_cluster in enumerate(self.num_cluster):
            # intialize faiss clustering parameters
            n, d = x.shape
            k = int(num_cluster)
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 20
            clus.nredo = 5
            clus.seed = seed
            clus.max_points_per_centroid = max(1000, int(3 * n / k))
            clus.min_points_per_centroid = max(10, int(n / k / 50))

            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = self.device
            index = faiss.GpuIndexFlatL2(res, d, cfg)

            with nostdout():
                clus.train(x_train.astype('float32'), index)

            D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
            idx2cluster = [int(n[0]) for n in I]

            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

            # sample-to-centroid distances for each cluster
            Dcluster = [[] for c in range(k)]
            for im, i in enumerate(idx2cluster):
                Dcluster[i].append(D[im][0])

            # concentration estimation (phi)
            density = np.zeros(k)
            for i, dist in enumerate(Dcluster):
                if len(dist) > 1:
                    d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                    density[i] = d

                    # if cluster only has one point, use the max to estimate its concentration
            dmax = density.max()
            for i, dist in enumerate(Dcluster):
                if len(dist) <= 1:
                    density[i] = dmax

            density = density.clip(np.percentile(density, 10),
                                   np.percentile(density, 90))  # clamp extreme values for stability
            density = self.temperature * density / density.mean()  # scale the mean to temperature

            centroids = torch.Tensor(centroids).cuda()
            centroids = nn.functional.normalize(centroids, p=2, dim=1)

            idx2cluster = torch.LongTensor(idx2cluster).cuda()
            density = torch.Tensor(density).cuda()

            results['centroids'].append(centroids)
            results['density'].append(density)
            results['idx2cluster'].append(idx2cluster)

        return results
