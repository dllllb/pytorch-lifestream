import torch

from ptls.metric_learn.dataset import collate_splitted_rows


def test_collate_splitted_row2():
    batch = [
        [
            ({'a': torch.tensor([0, 1, 2, 3]), 'b': torch.tensor([0, 1, 2, 3])}, 0),
            ({'a': torch.tensor([2, 3]), 'b': torch.tensor([2, 3])}, 0),
        ],
        [
            ({'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([1, 2, 3])}, 1),
            ({'a': torch.tensor([0, 1, 2, 3, 5]), 'b': torch.tensor([0, 1, 2, 3, 5])}, 1),
        ],
        [
            ({'a': torch.tensor([0, 3]), 'b': torch.tensor([0, 3])}, 2),
            ({'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([1, 2, 3])}, 2),
        ],
    ]
    x, y = collate_splitted_rows(batch)

    expected_x = {
        'a': torch.tensor([
            [0, 1, 2, 3, 0],
            [2, 3, 0, 0, 0],
            [1, 2, 3, 0, 0],
            [0, 1, 2, 3, 5],
            [0, 3, 0, 0, 0],
            [1, 2, 3, 0, 0],
        ]),
        'b': torch.tensor([
            [0, 1, 2, 3, 0],
            [2, 3, 0, 0, 0],
            [1, 2, 3, 0, 0],
            [0, 1, 2, 3, 5],
            [0, 3, 0, 0, 0],
            [1, 2, 3, 0, 0],
        ])
    }
    expected_l = torch.IntTensor([4, 2, 3, 5, 2, 3])
    expected_y = torch.LongTensor([0, 0, 1, 1, 2, 2])

    assert torch.all(x.payload['a'] == expected_x['a'])
    assert torch.all(x.payload['b'] == expected_x['b'])
    assert all(x.seq_lens == expected_l)
    assert torch.all(y == expected_y)
