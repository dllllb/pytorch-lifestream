import torch

from ptls.trx_encoder import PaddedBatch


def test_padded_batch_example():
    data = PaddedBatch(
        payload=torch.tensor([
            [1, 2, 0, 0],
            [3, 4, 5, 6],
            [7, 8, 9, 0],
        ]),
        length=torch.tensor([2, 4, 3]),
    )

    # check shape
    torch.testing.assert_close(data.payload.size(), (3, 4))

    # get first transaction
    torch.testing.assert_close(data.payload[:, 0], torch.tensor([1, 3, 7]))

    # get last transaction
    torch.testing.assert_close(data.payload[torch.arange(3), data.seq_lens - 1], torch.tensor([2, 6, 9]))

    # get all transaction flatten
    torch.testing.assert_close(data.payload[data.seq_len_mask.bool()], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))


def test_padded_batch_mask_tensor_trx_embedding():
    data = PaddedBatch(torch.randn(4, 5, 3), torch.tensor([2, 5, 1, 3]))
    out = data.seq_len_mask
    exp = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
    ]).long()
    torch.testing.assert_close(out, exp)


def test_padded_batch_mask_tensor_numerical():
    data = PaddedBatch(torch.randn(4, 5), torch.tensor([2, 5, 3, 1]))
    out = data.seq_len_mask
    exp = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
    ]).long()
    torch.testing.assert_close(out, exp)


def test_padded_batch_mask_dict():
    data = PaddedBatch({'col1': torch.randn(4, 5), 'col2': torch.randn(4, 5)}, torch.tensor([4, 5, 1, 3]))
    out = data.seq_len_mask
    exp = torch.tensor([
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0]
    ]).long()
    torch.testing.assert_close(out, exp)
