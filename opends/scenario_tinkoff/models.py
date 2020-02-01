import logging
import os

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from scenario_tinkoff.data import StoriesDataset
from scenario_tinkoff.feature_preparation import get_embeddings

logger = logging.getLogger(__name__)


class RunningAverageItem:
    def __init__(self, alpha, name, init_value=0.0, pattern='{:f}'):
        self.alpha = alpha
        self.value = init_value
        self.name = name
        self.pattern = pattern

    def update(self, t):
        self.value = self.alpha * self.value + (1.0 - self.alpha) * t.item()

    def show(self):
        return self.pattern.format(self.name, self.value)


class MixedEmbedding(torch.nn.Module):
    def __init__(self, layers, item_count,
                 fixed_vector_size,
                 norm=False,
                 trans_use_force=False,
                 trans_skip=False,

                 ):
        super().__init__()

        self.eps = 1e-5

        self.layers = layers
        self.item_count = item_count
        self.fixed_vector_size = fixed_vector_size
        self.norm = norm
        self.trans_use_force = trans_use_force
        self.trans_skip = trans_skip

        self.nn_layers = []
        self._nn_parameters = torch.nn.ParameterList()
        self._nn_modules = torch.nn.ModuleList()

    def create_layers(self):
        self.nn_layers = []
        self._nn_parameters = torch.nn.ParameterList()
        self._nn_modules = torch.nn.ModuleList()

        for layer_label in self.layers:
            if layer_label == '1':
                self.nn_layers.append(None)

            if layer_label == 'F':
                p = torch.nn.Parameter(torch.randn(1, 1))
                self.nn_layers.append(p)
                self._nn_parameters.append(p)

            if layer_label == 'E':
                m = torch.nn.Embedding(
                    num_embeddings=self.item_count,
                    embedding_dim=1,
                    padding_idx=None,
                )
                self.nn_layers.append(m)
                self._nn_modules.append(m)

            if layer_label == 'T':
                m = torch.nn.Linear(self.fixed_vector_size, 1)
                self.nn_layers.append(m)
                self._nn_modules.append(m)

    @property
    def output_size(self):
        return len(self.nn_layers)

    def forward(self, fixed_vectors, item_id):
        v_item = []

        for layer_label, nn_layer in zip(self.layers, self.nn_layers):
            if layer_label == '1':
                v_item.append(torch.ones(len(item_id), 1, device=item_id.device).float())

            if layer_label == 'F':
                v_item.append(nn_layer.repeat(len(item_id), 1))

            if layer_label == 'E':
                v_item.append(nn_layer(item_id.long()))

            if layer_label == 'T':
                v_item.append(nn_layer(fixed_vectors))

        h_item = torch.cat(v_item, dim=1)

        return h_item


class StoriesRecModel(torch.nn.Module):
    def __init__(self,
                 user_layers, user_fixed_vector_size, user_encoder, df_users,
                 item_layers, item_fixed_vector_size, item_encoder, df_items,
                 config, device,
                 ):
        super().__init__()

        self.eps = 1e-5

        self.user_encoder = user_encoder
        self.df_users = df_users
        self.item_encoder = item_encoder
        self.df_items = df_items

        self.valid_fn = None
        self.config = config
        self.device = device

        self.embed_user = MixedEmbedding(
            layers=user_layers,
            item_count=len(self.user_encoder) + 1,
            fixed_vector_size=user_fixed_vector_size,
            trans_skip=True,
        )

        self.embed_item = MixedEmbedding(
            layers=item_layers,
            item_count=len(self.item_encoder) + 1,
            fixed_vector_size=item_fixed_vector_size,
        )

        self.embed_user.create_layers()
        self.embed_item.create_layers()

        self.final_bias = torch.nn.Parameter(torch.randn(1))

    def add_valid_fn(self, valid_fn):
        self.valid_fn = valid_fn

    def forward(self, t_users, user_id, t_items, item_id):
        B = user_id.size()[0]

        h_user = self.embed_user(t_users, user_id)
        h_item = self.embed_item(t_items, item_id)

        hidden = (h_user * h_item).sum(dim=1) + self.final_bias.repeat(B)
        return hidden

    def batch_predict(self, t_users, user_id, t_items, item_id, drop_mask, k):
        h_user = self.embed_user(t_users, user_id)
        h_item = self.embed_item(t_items, item_id)

        user_item = torch.mm(h_user, torch.transpose(h_item, 0, 1))
        user_item = self.activation(user_item)

        _minus_one = torch.ones(user_item.size()).to(user_item.device).float() * -1.0
        user_item = drop_mask * _minus_one + (1.0 - drop_mask) * user_item
        _, user_item = torch.topk(user_item, k=k, dim=1)
        return user_item

    def _get_loss(self):
        loss_name = self.config.loss
        if loss_name == 'ranking':
            loss_fn = MultiClassMarginRankingLoss(margin=self.config.loss_margin)
        elif loss_name == 'bce':
            loss_fn = torch.nn.BCELoss()
        elif loss_name == 'mae':
            loss_fn = torch.nn.L1Loss()
        elif loss_name == 'mse':
            loss_fn = torch.nn.MSELoss()
        else:
            raise AttributeError(f'Unknown loss: {loss_name}')
        logger.info(f'Used {loss_fn.__class__.__name__}')
        return loss_fn

    def _get_optim(self, lr=None):
        if lr is None:
            lr = self.config.optim_lr

        logger.info(f'Used model "{self.__class__.__name__}" with parameters: ' +
                    ', '.join([f'{n}: {p.size()}' for n, p in self.named_parameters()]))

        if len(self.config.optim_weight_decay) > 1:
            logger.info('Used decays [{}]'.format(
                ", ".join(f'{k}: {decay}' for (k, v), decay in zip(self.named_parameters(),
                                                                   self.config.optim_weight_decay))
            ))

            parameters = [{
                'params': [v],
                'weight_decay': decay
            }
                for (k, v), decay in zip(self.named_parameters(), self.config.optim_weight_decay)
            ]

            return torch.optim.Adam(parameters, lr=lr)
        else:
            logger.info(f'Used decay = {self.config.optim_weight_decay[0]} for all layers')

            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.config.optim_weight_decay[0])

    def _get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=self.config.lr_step_size,
                                               gamma=self.config.lr_step_gamma)

    def _get_train_data_loader(self, df_log):
        return DataLoader(
            dataset=StoriesDataset(
                df_log=df_log,
                df_user=self.df_users, user_encoder=self.user_encoder,
                df_item=self.df_items, item_encoder=self.item_encoder,
            ),
            batch_size=self.config.train_batch_size,
            num_workers=self.config.train_num_workers)

    def _get_valid_data_loader(self, df_log):
        return DataLoader(
            dataset=StoriesDataset(
                df_log=df_log,
                df_user=self.df_users, user_encoder=self.user_encoder,
                df_item=self.df_items, item_encoder=self.item_encoder,
            ),
            batch_size=self.config.valid_batch_size,
            shuffle=False,
            num_workers=self.config.valid_num_workers)

    def model_train(self, df_log):
        self.to(self.device)
        self.train()

        data_loader = self._get_train_data_loader(df_log)
        optimiser = self._get_optim()
        scheduler = self._get_scheduler(optimiser)
        loss_fn = self._get_loss()

        avg_loss = RunningAverageItem(0.95, 'loss', 0.0, '{}: [{:.4f}]')
        avg_min_out = RunningAverageItem(0.95, 'avg_min_out', 0.0, '{}: [{:.2f}]')
        avg_max_out = RunningAverageItem(0.95, 'avg_max_out', 1.0, '{}: [{:.2f}]')
        running_metrics = [avg_loss, avg_min_out, avg_max_out]
        metrics = []

        max_epoch = self.config.max_epoch
        for n_epoch in range(1, max_epoch + 1):
            epoch_loss = 0.0

            progress_desc = ', '.join(
                [f'Epoch [{n_epoch:2d}/{max_epoch}]'] +
                [m.show() for m in running_metrics]
            )

            with tqdm(total=len(data_loader), desc=progress_desc,
                      mininterval=1.0) as progress:
                for batch in data_loader:
                    x_user, x_user_id, x_item, x_item_id, target = (t.to(self.device) for t in batch)

                    optimiser.zero_grad()
                    output = self(x_user, x_user_id, x_item, x_item_id)

                    avg_min_out.update(output.min())
                    avg_max_out.update(output.max())

                    loss = loss_fn(output, target)
                    avg_loss.update(loss)
                    epoch_loss += loss.item()

                    loss.backward()
                    optimiser.step()

                    progress.update(1)
                    progress_desc = ', '.join(
                        [f'Epoch [{n_epoch:2d}/{max_epoch}]'] +
                        [m.show() for m in running_metrics]
                    )
                    progress.set_description(progress_desc)

                scheduler.step()

                epoch_loss = epoch_loss / len(data_loader)
                progress_desc = ', '.join(
                    [f'Epoch [{n_epoch:2d}/{max_epoch}]'] +
                    [m.show() for m in running_metrics] +
                    [
                        f'epoch_loss [{epoch_loss:.4f}]',
                    ])
                progress.set_description(progress_desc)

            if self.valid_fn is not None:
                score = self.valid_fn()
                for k, v in score.items():
                    logger.info(f'Score after epoch [{n_epoch:2d}/{max_epoch}] {k}: {v:.4f}')
                scores = [(k, v) for k, v in score.items()]
            else:
                scores = []

            metrics.append({n: float(v) for n, v in
                            [
                                ('epoch', n_epoch),
                                ('epoch_loss', epoch_loss),
                            ] + [(m.name, m.value) for m in running_metrics] + scores})
        return metrics

    def model_predict(self, df_log_predict):
        self.to(self.device)
        self.eval()

        data_loader = self._get_valid_data_loader(df_log_predict)

        s_predicted_relevance = []

        with tqdm(total=len(data_loader), desc='Predict',
                  mininterval=1.0) as progress:
            for batch in data_loader:
                x_user, x_user_id, x_item, x_item_id, _ = (t.to(self.device) for t in batch)
                with torch.no_grad():
                    output = self(x_user, x_user_id, x_item, x_item_id)
                s_predicted_relevance.append(output.cpu().numpy())

                progress.update(1)

        df_scores = df_log_predict.copy()
        df_scores['relevance'] = np.concatenate(s_predicted_relevance)

        return df_scores

    def model_predict_top_k(self, df_log_exclude, id_users, id_items, k):
        df_log_exclude = df_log_exclude.groupby(['user_id', 'item_id'])[['relevance']].count().reset_index()

        s_predicted_users = []
        s_predicted_items = []

        self.to(self.device)
        self.eval()

        df_users = self.df_users.reindex(index=id_users)
        df_items = self.df_items.reindex(index=id_items)
        logger.info(f'Predict for {len(df_users)} users and {len(df_items)} items')

        t_items = torch.from_numpy(df_items.values.astype(np.float32))
        t_items = t_items.to(self.device)

        t_item_id = torch.from_numpy(np.array([self.item_encoder.get(item_id, 0)
                                               for item_id in id_items]).astype(np.int16))
        t_item_id = t_item_id.to(self.device)

        a_items = np.array(id_items)

        valid_batch_size = self.config.valid_batch_size
        for ix_start in tqdm(range(0, len(df_users), valid_batch_size), desc='Predict'):
            ix_end = ix_start + valid_batch_size
            df_batch_users = df_users.iloc[ix_start:ix_end]
            t_batch_users = torch.from_numpy(df_batch_users.values.astype(np.float32))
            t_batch_users = t_batch_users.to(self.device)
            batch_id_users = df_batch_users.index.tolist()

            t_user_id = torch.from_numpy(
                np.array([self.user_encoder.get(user_id, 0) for user_id in batch_id_users]).astype(np.int32))
            t_user_id = t_user_id.to(self.device)

            df_log_e = df_log_exclude[df_log_exclude['user_id'].isin(batch_id_users)]
            if len(df_log_e) == 0:
                t_drop_mask = np.zeros((len(batch_id_users), len(id_items)), dtype=np.int32)
            else:
                df_log_e = df_log_e.pivot(index='user_id', columns='item_id', values='relevance')
                df_log_e = df_log_e.reindex(index=batch_id_users, columns=id_items)
                t_drop_mask = (~df_log_e.isna()).astype(np.int32).values
            t_drop_mask = torch.from_numpy(t_drop_mask)
            t_drop_mask = t_drop_mask.to(self.device).float()

            with torch.no_grad():
                t_predict = self.batch_predict(t_batch_users, t_user_id, t_items, t_item_id, t_drop_mask, k)

            nn_predict = t_predict.cpu().numpy()
            s_predicted_users.extend((user_id for user_id in batch_id_users for _ in range(k)))
            item_indexes = nn_predict.flatten()
            s_predicted_items.extend(a_items[item_indexes].tolist())

        return pd.DataFrame({'user_id': s_predicted_users, 'item_id': s_predicted_items})


class PopularModel:
    def __init__(self, device, config, alpha=1000, beta=1000):
        self.device = device
        self.config = config
        self.alpha = alpha
        self.beta = beta

        self.items_popularity = None
        self.avg_num_items = None

    def model_train(self, df_log):
        logger.info(f'Used model "{self.__class__.__name__}"')

        if self.config.use_low_freq_train_items:
            df_user_items = df_log['user_id'].value_counts()
            _user_total = len(df_user_items)
            low_freq_users = df_user_items.index.tolist()[int(_user_total * 0.95):]
            df_log = df_log[df_log['user_id'].isin(low_freq_users)]
            logger.info(f'use_low_freq_train_items: {_user_total} users was reduced to {len(low_freq_users)}')

        if self.config.use_mean_relevance:
            df_popularity = df_log.groupby(['user_id', 'item_id'])[['relevance']].mean()
            df_popularity = df_popularity[df_popularity['relevance'].gt(0.0)].reset_index()
            df_popularity = df_popularity.groupby('item_id')['relevance'].sum()
            logger.info(f'use_mean_relevance')

        elif self.config.use_item_freq_predict:
            df_popularity = df_log.groupby('item_id')['relevance'].mean()
            logger.info(f'use_item_freq_predict')

        else:
            df_popularity = df_log[df_log['relevance'].eq(1)]
            df_popularity = df_popularity.groupby('item_id')['user_id'].count()
            logger.info(f'count relevance=1')

        self.items_popularity = df_popularity

        # считаем среднее кол-во просмотренных items у каждого user
        self.avg_num_items = np.ceil(df_log.groupby('user_id')['item_id'].count().mean())
        logger.info(f"Mean item count per user: {self.avg_num_items}")

        metrics = [
            {
                'item_count': len(self.items_popularity),
                'avg_num_items': self.avg_num_items,
            },
        ]
        return metrics

    def _get_item_scores(self):
        df_items_to_rec = self.items_popularity
        count_sum = df_items_to_rec.sum()
        df_items_to_rec = (df_items_to_rec + self.alpha) / (count_sum + self.beta)
        return df_items_to_rec

    def model_predict(self, df_log_exclude, df_log_predict):
        logger.info('Predict start')
        df_scores = df_log_predict.drop(columns='relevance').copy()
        item_scores = self._get_item_scores()
        df_scores = pd.merge(df_scores, item_scores.rename('relevance'),
                             how='left', left_on='item_id', right_index=True)
        df_scores['relevance'] = df_scores['relevance'].fillna(0.0)

        if self.config.exclude_seen_items:
            logger.info('exclude_seen_items start')
            df_scores = pd.merge(
                df_scores,
                df_log_exclude.drop(columns='relevance').assign(exclude_x=-1),
                how='left', on=['user_id', 'item_id'])
            df_scores['exclude_x'] = df_scores['exclude_x'].fillna(0)
            df_scores['relevance'] = df_scores['relevance'] + df_scores['exclude_x']
            del df_scores['exclude_x']

        logger.info('Predict done')
        return df_scores

    def model_predict_top_k(
            self,
            df_log_exclude,
            id_users, id_items,
            k,
    ):
        df_log_exclude = df_log_exclude.groupby(['user_id', 'item_id'])[['relevance']].count().reset_index()

        s_predicted_users = []
        s_predicted_items = []

        logger.info(f'Predict for {len(id_users)} users and {len(id_items)} items')

        df_items_to_rec = self._get_item_scores()
        df_items_to_rec = df_items_to_rec.reindex(index=id_items).fillna(0.0)

        id_items = df_items_to_rec.sort_values(ascending=False).iloc[:int(k + self.avg_num_items)].index.tolist()
        df_items_to_rec = df_items_to_rec.loc[id_items]

        a_items = np.array(id_items)
        a_users = np.array(id_users)

        item_predict_row = torch.from_numpy(df_items_to_rec.values.astype(np.float32).reshape(1, -1)).to(self.device)
        minus_one_row = torch.ones(1, len(id_items), device=self.device).float() * -1.0

        valid_batch_size = self.config.valid_batch_size
        for ix_start in tqdm(range(0, len(id_users), valid_batch_size), desc='Predict'):
            ix_end = ix_start + valid_batch_size
            batch_id_users = a_users[ix_start:ix_end]

            df_log_e = df_log_exclude[df_log_exclude['user_id'].isin(batch_id_users)]
            if len(df_log_e) == 0:
                t_drop_mask = np.zeros((len(batch_id_users), len(id_items)), dtype=np.int32)
            else:
                df_log_e = df_log_e.pivot(index='user_id', columns='item_id', values='relevance')
                df_log_e = df_log_e.reindex(index=batch_id_users, columns=id_items)
                t_drop_mask = (~df_log_e.isna()).astype(np.int32).values
            t_drop_mask = torch.from_numpy(t_drop_mask)
            t_drop_mask = t_drop_mask.to(self.device).float()

            user_item = item_predict_row.expand(len(batch_id_users), item_predict_row.size()[1])
            _minus_one = minus_one_row.expand(len(batch_id_users), minus_one_row.size()[1])
            user_item = t_drop_mask * _minus_one + (1.0 - t_drop_mask) * user_item

            _, t_predict = torch.topk(user_item, k=k, dim=1)

            nn_predict = t_predict.cpu().numpy()
            item_indexes = nn_predict.flatten()
            s_predicted_users.extend((user_id for user_id in batch_id_users for _ in range(nn_predict.shape[1])))
            s_predicted_items.extend(a_items[item_indexes].tolist())

        return pd.DataFrame({'user_id': s_predicted_users, 'item_id': s_predicted_items})


class PairwiseMarginRankingLoss(torch.nn.Module):
    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean'):
        """
        Pairwise Margin Ranking Loss. All setted parameters redirected to nn.MarginRankingLoss.
        All the difference is that pairs automatically generated for margin ranking loss.
        All possible pairs of different class are generated.
        """
        super().__init__()
        self.margin_loss = torch.nn.MarginRankingLoss(margin, size_average, reduce, reduction)

    def forward(self, prediction, label):
        """
        Get pairwise margin ranking loss.
        :param prediction: tensor of shape Bx1 of predicted probabilities
        :param label: tensor of shape Bx1 of true labels for pair generation
        """

        # positive-negative selectors
        mask_0 = label == 0.0
        mask_1 = label == 1.0

        # selected predictions
        pred_0 = torch.masked_select(prediction, mask_0)
        pred_1 = torch.masked_select(prediction, mask_1)
        pred_1_n = pred_1.size()[0]
        pred_0_n = pred_0.size()[0]

        if pred_1_n > 0 and pred_0_n:
            # create pairs
            pred_00 = pred_0.unsqueeze(0).repeat(1, pred_1_n)
            pred_11 = pred_1.unsqueeze(1).repeat(1, pred_0_n).view(pred_00.size())
            out01 = -1 * torch.ones(pred_1_n * pred_0_n).to(prediction.device)

            return self.margin_loss(pred_00.view(-1), pred_11.view(-1), out01)
        else:
            return torch.sum(prediction) * 0.0


class MultiClassMarginRankingLoss(torch.nn.Module):
    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean'):
        """
        Pairwise Margin Ranking Loss. All setted parameters redirected to nn.MarginRankingLoss.
        All the difference is that pairs automatically generated for margin ranking loss.
        All possible pairs of different class are generated.
        """
        super().__init__()
        self.margin_loss = torch.nn.MarginRankingLoss(margin, size_average, reduce, reduction)

    def forward(self, prediction, label):
        """
        Get pairwise margin ranking loss.
        :param prediction: tensor of shape Bx1 of predicted probabilities
        :param label: tensor of shape Bx1 of true labels for pair generation
        """

        # only pairs which can be ranked
        mask = (label.unsqueeze(0).repeat(len(label), 1) - label.unsqueeze(1).repeat(1, len(label)) > 0).int()
        ix_pairs = mask.nonzero()
        if len(ix_pairs) > 0:
            out01 = -1 * torch.ones(len(ix_pairs)).to(prediction.device)
            return self.margin_loss(
                prediction[ix_pairs[:, 0]],
                prediction[ix_pairs[:, 1]],
                out01,
            )
        else:
            return torch.sum(prediction) * 0.0


class ALSModel:
    """
2019-11-26 16:36:05,983 INFO    <module>        : ALSModel predict score_all: 0.5528
2019-11-26 16:36:05,983 INFO    <module>        : ALSModel predict score_cold: 0.5400
2019-11-26 16:36:05,984 INFO    <module>        : ALSModel predict score_warm: 0.5647
    """

    def __init__(self, factors, device, config):
        import implicit

        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        self.model = implicit.als.AlternatingLeastSquares(factors=factors)
        self.device = device
        self.config = config

        self.df_item_freq = None
        self.item_index = None
        self.user_index = None
        self.item_map = None
        self.user_map = None

    def get_item_user_matrix(self, df_log):
        df = df_log[['user_id', 'item_id', 'relevance']].copy()

        if self.config.pop_norm:
            df = pd.merge(df, self.df_item_freq.rename('alpha'), left_on='item_id', right_index=True)
            df['relevance'] = df['relevance'] - df['alpha']
            del df['alpha']
        else:
            df['relevance'] = df['relevance'].map({0: -1, 1: 1})

        df = df.groupby(['user_id', 'item_id'])[['relevance']].mean().reset_index()
        df['user_id'] = df['user_id'].map(self.user_map)
        df['item_id'] = df['item_id'].map(self.item_map)
        df = df[~df.isna().any(axis=1)]
        return scipy.sparse.csr_matrix((df['relevance'],
                                        (df['item_id'],
                                         df['user_id']),
                                        ),
                                       shape=(len(self.item_index), len(self.user_index)))

    def model_train(self, df_log):
        self.df_item_freq = df_log.groupby('item_id')['relevance'].mean()

        self.item_index = df_log['item_id'].unique()
        self.user_index = df_log['user_id'].unique()
        self.item_map = {k: v for v, k in enumerate(self.item_index)}
        self.user_map = {k: v for v, k in enumerate(self.user_index)}

        item_user_data = self.get_item_user_matrix(df_log)
        logger.info(f'item_user_data matrix of type {type(item_user_data)} with shape: {item_user_data.shape}')
        self.model.fit(item_user_data)

    def model_predict(self, df_log_exclude, df_log_predict):
        user_item_data = self.get_item_user_matrix(df_log_exclude).T.tocsr()

        predicted_rows = []

        for_predict = df_log_predict.groupby('user_id')['item_id'].apply(np.array)
        for user_id, item_list in tqdm(for_predict.items(), total=len(for_predict), desc='Predict'):
            user_ix = self.user_map.get(user_id, -1)
            if user_ix == -1:
                for item_id in item_list:
                    predicted_rows.append((user_id, item_id, 0.0))
            else:
                item_ix = [self.item_map.get(item_id) for item_id in item_list if item_id in self.item_map]
                if len(item_ix) > 0:
                    als_predict = self.model.rank_items(user_ix, user_item_data, item_ix)
                    for item_id, score in als_predict:
                        predicted_rows.append((user_id, self.item_index[item_id], score))
                for item_id in [item_id for item_id in item_list if item_id not in self.item_map]:
                    predicted_rows.append((user_id, item_id, 0.0))

        predicted_rows = pd.DataFrame(predicted_rows, columns=['user_id', 'item_id', 'relevance'])
        predicted = pd.merge(
            df_log_predict.drop(columns='relevance'),
            predicted_rows,
            how='inner', on=['user_id', 'item_id'],
        )

        if self.config.pop_norm:
            predicted = pd.merge(predicted, self.df_item_freq.rename('alpha'), left_on='item_id', right_index=True,
                                 how='left')
            predicted['alpha'] = predicted['alpha'].fillna(0.0)
            predicted['relevance'] = predicted['relevance'] + predicted['alpha']
            del predicted['alpha']
        else:
            predicted['relevance'] = (predicted['relevance'] + 1.0) / 2.0

        return predicted

    def model_predict_top_k(
            self,
            df_log_exclude,
            id_users, id_items,
            k,
    ):
        raise NotImplementedError()


class RandomModel:
    """
2019-11-26 16:52:56,702 INFO    <module>        : RandomModel predict score_all: 0.5483
2019-11-26 16:52:56,703 INFO    <module>        : RandomModel predict score_cold: 0.5413
2019-11-26 16:52:56,703 INFO    <module>        : RandomModel predict score_warm: 0.5548
    """

    def __init__(self, device, config):
        self.device = device
        self.config = config

    def model_train(self, df_log):
        pass

    def model_predict(self, df_log_exclude, df_log_predict):
        predicted = df_log_predict.copy()
        predicted['relevance'] = np.random.rand(len(predicted))
        return predicted

    def model_predict_top_k(
            self,
            df_log_exclude,
            id_users, id_items,
            k,
    ):
        raise NotImplementedError()


def model_inspection(model):
    print('--- Inspect model: ---')
    for k, v in model.named_parameters():
        v = v.detach().cpu().numpy()
        print(f'{k:40}: ', end='')

        if v.size <= 10:
            print(v)
        else:
            values_str = [f'{p:7.3f}' for p in np.percentile(v, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])]
            print('p: % {} %'.format(" ".join(values_str)))
