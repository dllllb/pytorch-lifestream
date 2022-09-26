import torch
from ptls.frames.bert import MlmDataset


class TabformerDataset(MlmDataset):
    pass


class TabformerIterableDataset(TabformerDataset, torch.utils.data.IterableDataset):
    pass
