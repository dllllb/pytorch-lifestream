# `ptls.trx_encoder`
All classes from `ptls.nn.trx_encoder` also available in `ptls.nn`

`ptls.nn.trx_encoder` helps to make a representation for single transactions.

## `ptls.nn.TrxEncoder`

Now we have an input data:
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
And we can define a TrxEncoder
```python
model = TrxEncoder(
    embeddings={
        'mcc_code': {'in': 10, 'out': 6},
        'currency': {'in': 4, 'out': 2},
    },
    numeric_values={'amount': 'identity'},
)
```
We should provide feature description to `TrxEncoder`.
Dictionary size and embedding size for categorical features. Scaler name for numerical features.
`identity` means no rescaling.

`TrxEncoder` concatenates all feature embeddings, sow output embedding size will be `6 + 2 + 1`.
You may get output size from `TrxEncoder` with property:
```python
>>> model.output_size
Out[]: 6
```

Let's transform our features to embeddings
```python
z = model(x)
```

`z` is also `PaddedBatch`. `z.seq_lens` equals `x.seq_lens`.
`z.payload` isn't dict, it's tensor of shape (B, T, H). In our example `B, T = 3, 8` is input feature shape,
`H = 6` is output size of model.

Now we can use other layers which consume transactional embeddings.


## Classes
See docstrings for classes:

- `ptls.nn.TrxEncoder`
