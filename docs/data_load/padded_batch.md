# `ptls.data_load.padded_batch.PaddedBatch`

Input data is a raw feature formats. You can transform your transaction to correct format with `ptls.data` module.
Common description or sequential data and used data formats are [here](../sequential_data_definition.md)
Input data are covered in `ptls.data_load.padded_batch.PaddedBatch` class.

We can create `PaddedBatch` object manually for demo and test purposes.

```python
x = PaddedBatch(
    payload={
        'mcc_code': torch.randint(1, 10, (3, 8)),
        'currency': torch.randint(1, 4, (3, 8)),
        'amount': torch.randn(3, 8) * 4 + 5,
    },
    length=torch.Tensor([2, 8, 5]).long()
)
```

Here `x` contains three features. Two are categorical and one is numerical:

- `mcc_code` is categorical with `dictionary_size=10`
- `currency` is categorical with `dictionary_size=4`
- `amount` is numerical with `mean=5` and `std=4`

`x` contains 5 sequences with `maximum_length=12`. Real lengths of each sequence are `[2, 8, 5]`.

We can access `x` content via `PaddedBatch` properties `x.payload` and `x.seq_lens`.

Real data have sequences are padded with zeros. We can imitate it with `x.seq_len_mask`. 
It returns tensor with 1 if a position inside corresponded seq_len and 0 if position outside.
Let's check out example
```python
>>> x.seq_len_mask
Out: 
tensor([[1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0]])
```
There are 2, 8 and 5 valid tokens in lines.

More way of `seq_len_mask` usage are in `PaddedBatch` docstring.

We can recreate our `x` with modified content:
```python
x = PaddedBatch({k: v * x.seq_len_mask for k, v in x.payload.items()}, x.seq_lens)
```

Now we can check `x.payload` and see features looks like real padded data:
```python
>>> x.payload['mcc_code']
Out: 
tensor([[8, 1, 0, 0, 0, 0, 0, 0],
        [5, 5, 9, 9, 4, 9, 3, 1],
        [4, 2, 2, 3, 3, 0, 0, 0]])
```

All invalid tokens are replaced with zeros.

Generally, all layers respect `PaddedBatch.seq_lens` and no explicit zeroing of padded characters is required.

## Classes
See docstrings for classes:

- `ptls.data_load.padded_batch.PaddedBatch`
