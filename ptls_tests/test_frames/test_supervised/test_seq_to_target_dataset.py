import torch
from ptls.frames.supervised.seq_to_target_dataset import SeqToTargetDataset


def test_seq_to_target_dataset_default_long():
    dataset = SeqToTargetDataset([{
        'mcc_code': torch.randint(1, 10, (seq_len,)),
        'amount': torch.randn(seq_len),
        'event_time': torch.arange(seq_len),  # shows order between transactions
        'target': target,
    } for seq_len, target in zip(
        torch.randint(100, 200, (4,)),
        [0, 0, 1, 1],
    )], target_col_name='target')
    dl = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=dataset.collate_fn)
    x, y = next(iter(dl))
    torch.testing.assert_close(y, torch.LongTensor([0, 0, 1, 1]))

def test_seq_to_target_dataset_default_double():
    dataset = SeqToTargetDataset([{
        'mcc_code': torch.randint(1, 10, (seq_len,)),
        'amount': torch.randn(seq_len),
        'event_time': torch.arange(seq_len),  # shows order between transactions
        'target': target,
    } for seq_len, target in zip(
        torch.randint(100, 200, (4,)),
        [0.1, 0.4, 1.0, 0.9],
    )], target_col_name='target')
    dl = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=dataset.collate_fn)
    x, y = next(iter(dl))
    torch.testing.assert_close(y, torch.FloatTensor([0.1, 0.4, 1.0, 0.9]))


def test_seq_to_target_dataset_type_double():
    dataset = SeqToTargetDataset([{
        'mcc_code': torch.randint(1, 10, (seq_len,)),
        'amount': torch.randn(seq_len),
        'event_time': torch.arange(seq_len),  # shows order between transactions
        'target': target,
    } for seq_len, target in zip(
        torch.randint(100, 200, (4,)),
        [0, 0, 1, 1],
    )], target_col_name='target', target_dtype=torch.double)
    dl = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=dataset.collate_fn)
    x, y = next(iter(dl))
    torch.testing.assert_close(y, torch.DoubleTensor([0.0, 0.0, 1.0, 1.0]))

def test_seq_to_target_dataset_type_array():
    dataset = SeqToTargetDataset([{
        'mcc_code': torch.randint(1, 10, (seq_len,)),
        'amount': torch.randn(seq_len),
        'event_time': torch.arange(seq_len),  # shows order between transactions
        'target': torch.randn(2, 8),
    } for seq_len in torch.randint(100, 200, (1000,))], target_col_name='target')
    dl = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=dataset.collate_fn)
    x, y = next(iter(dl))
    assert y.size() == (10, 2, 8)
    assert y.dtype == torch.float32
