import logging
import torch

logger = logging.getLogger(__name__)


class CPCShellV2(torch.nn.Module):
    def __init__(self, encoder, embedding_size, k_pos_samples):
        super().__init__()

        self.encoder = encoder
        self.k_pos_samples = k_pos_samples

        history_size = k_pos_samples - 1  # predict one last sample based on all previous
        self.linear_predictor = torch.nn.Linear(embedding_size * history_size, embedding_size)

    def forward(self, x):
        z = self.encoder(x)
        return z


class CPCLossV2(torch.nn.Module):
    def __init__(self, k_pos_samples, m_neg_samples, linear_predictor):
        super(CPCLossV2, self).__init__()

        self.k_pos_samples = k_pos_samples
        self.m_neg_samples = m_neg_samples

        self.linear_predictor = linear_predictor

    def forward(self, embeddings, target):
        embeddings = embeddings

        k_pos_samples = self.k_pos_samples
        n = embeddings.size()[0] // k_pos_samples
        h = embeddings.size()[1]
        m_neg_samples = min(self.m_neg_samples, k_pos_samples * (n - 1))

        # assert m_neg_samples <= (n - 1) * k_pos_samples, (m_neg_samples, (n - 1) * k_pos_samples)

        # pos pred
        history_x_indexes = (torch.arange(n * k_pos_samples) + 1) % k_pos_samples != 0
        history_y_indexes = (torch.arange(n * k_pos_samples) + 1) % k_pos_samples == 0

        hist_x = embeddings[history_x_indexes].reshape(n, -1)  # shape: [n, embedding_size * (k - 1)]
        hist_y = embeddings[history_y_indexes]

        predicts = self.linear_predictor(hist_x)
        positive_pred_logit = predicts.mul(hist_y).sum(axis=-1)

        # negatives
        x = ((target.expand(n * k_pos_samples, n * k_pos_samples) -
              target.expand(n * k_pos_samples, n * k_pos_samples).t()) != 0).nonzero(as_tuple=False)[:, 1]
        neg_samples = x.view(n, k_pos_samples, -1)[:, 0, :]
        perm_ix = torch.cat(
            [torch.stack([torch.ones(m_neg_samples).long() * i,
                          torch.randperm(k_pos_samples * (n - 1))[:m_neg_samples]]).t() for i in range(n)])
        neg_embed = embeddings[neg_samples[perm_ix[:, 0], perm_ix[:, 1]]].view(n, m_neg_samples, h)

        neg_logit = (predicts.unsqueeze(1).repeat(1, m_neg_samples, 1) * neg_embed).sum(-1)

        loss = torch.nn.functional.log_softmax(
            torch.cat([positive_pred_logit.unsqueeze(-1), neg_logit], dim=-1),
            dim=-1)[:, 0]
        return -1.0 * loss.mean(), None
