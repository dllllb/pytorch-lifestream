import functools
import operator
import torch

from dltranz.data_load import padded_collate_wo_target


class SentencePairsHead(torch.nn.Module):
    def __init__(self, base_model, embeds_dim, config):
        super().__init__()
        self.base_model = base_model
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embeds_dim * 2, config['hidden_size'], bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(config['drop_p']),
            torch.nn.BatchNorm1d(config['hidden_size']),
            torch.nn.Linear(config['hidden_size'], 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        left, right = x
        x = torch.cat([self.base_model(left), self.base_model(right)], dim=1)
        return self.head(x).squeeze(-1)


def collate_sop_pairs(batch):
    batch = functools.reduce(operator.iadd, batch)

    targets = torch.randint(low=0, high=2, size=(len(batch),))

    lefts = [left if target else right for (left, right), target in zip(batch, targets)]
    rights = [right if target else left for (left, right), target in zip(batch, targets)]

    return (
        (
            padded_collate_wo_target(lefts),
            padded_collate_wo_target(rights)
        ),
        targets
    )
