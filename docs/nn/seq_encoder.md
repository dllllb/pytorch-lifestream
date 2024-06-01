# `ptls.nn.seq_encoder`
All classes from `ptls.nn.seq_encoder` also available in `ptls.nn`

`ptls.nn.trx_encoder` works with individual transaction.
`ptls.nn.seq_encoder` takes into account sequential structure and the links between transactions.

There are 2 types of seq encoders:

- required embeddings as input
- requires raw features as input

## Embeddings as input

We implement ptls-api for `torch` and `huggingface` sequential layers:

- `ptls.nn.RnnEncoder` for `torch.nn.GRU`
- `ptls.nn.TransformerEncoder` for `torch.nn.TransformerEncoder`
- `ptls.nn.LongformerEncoder` for `transformers.LongformerModel`

They expect vectorized input, which can be obtained with `TrxEncoder`.

Output format controlled by `is_reduce_sequence` property. `True` means that sequence will be reduced 
to one single vector.  It's last hidden state for RNN and CLS token output for transformer.
`False` means than all hidden vectors for all transactions will be returned.  Set this property based on your needs.
It's possible to set it during encoder initialisation. It's possible to change it in runtime.

Simple Example:
```python
x = PaddedBatch(torch.randn(10, 80, 4), torch.randint(40, 80, (10,)))
seq_encoder = RnnEncoder(input_size=4, hidden_size=16)
y = seq_encoder(x)
assert y.payload.size() == (10, 80, 16)
```

More complicated example:
```python
x = PaddedBatch(
    payload={
        'mcc_code': torch.randint(1, 10, (3, 8)),
        'currency': torch.randint(1, 4, (3, 8)),
        'amount': torch.randn(3, 8) * 4 + 5,
    },
    length=torch.Tensor([2, 8, 5]).long()
)

trx_encoder = TrxEncoder(
    embeddings={
        'mcc_code': {'in': 10, 'out': 6},
        'currency': {'in': 4, 'out': 2},
    },
    numeric_values={'amount': 'identity'},

)
seq_encoder = RnnEncoder(input_size=trx_encoder.output_size, hidden_size=16)

z = trx_encoder(x)
y = seq_encoder(z)  # embeddings for each transaction
seq_encoder.is_reduce_sequence = True
h = seq_encoder(z)  # embeddings for sequences, aggregate all transactions in one embedding

assert y.payload.size() == (3, 8, 16)
assert h.size() == (3, 16)
```

Usually `seq_encoder` is used with preliminary `trx_encoder`. It's possible to pack them to `torch.nn.Sequential`.

It's possible to add more layers between `trx_encoder` and `seq_encoder` (linear, normalisation, convolutions, ...). 
They should work with PaddedBatch. Examples will be presented later. Such layers also works after `seq_encoder`
with `is_reduce_sequence=False`.


## Features as input

As you can see `TrxEncoder` works with raw features and compatible with embedding seq encoder.
We make a composition layers, which contains `TrxEncoder` and one `SeqEncoder` implementation.
There are:

- `ptls.nn.RnnSeqEncoder` with `RnnEncoder`
- `ptls.nn.TransformerSeqEncoder` with `TransformerEncoder`
- `ptls.nn.LongformerSeqEncoder` with `LongformerEncoder`

They work as simple `Sequential(trx_encoder, seq_encoder)` and support `is_reduce_sequence` property.
The main advantage that you can simply create such encoder from config file using `hydra instantiate` tools.
You can avoid of explicit set of `seq_encoder.input_size`, they will be taken from `trx_encoder`.  Let's compare.

Sequential-style:
```python
config = """
    model:
        _target_: torch.nn.Sequential
        _args_:
        - 
            _target_: ptls.nn.TrxEncoder
            embeddings:
                mcc_code:
                    in: 10
                    out: 6
                currency:
                    in: 4
                    out: 2
            numeric_values:
                amount: identity
        -
            _target_: ptls.nn.RnnEncoder
            input_size: 9  # depends on TrxEncoder output
            hidden_size: 24
"""
model = hydra.utils.instantiate(OmegaConf.create(config))['model']
```

SeqEncoder-style:
```python
config = """
    model:
        _target_: ptls.nn.RnnSeqEncoder
        trx_encoder:
            _target_: ptls.nn.TrxEncoder
            embeddings:
                mcc_code:
                    in: 10
                    out: 6
                currency:
                    in: 4
                    out: 2
            numeric_values:
                amount: identity
        hidden_size: 24
"""
model = hydra.utils.instantiate(OmegaConf.create(config))['model']
```

The second config is simpler. Both of configs make an identical model. You can check:
```python
x = PaddedBatch(
    payload={
        'mcc_code': torch.randint(1, 10, (3, 8)),
        'currency': torch.randint(1, 4, (3, 8)),
        'amount': torch.randn(3, 8) * 4 + 5,
    },
    length=torch.Tensor([2, 8, 5]).long()
)

y = model(x)
```

## AggFeatureSeqEncoder

`ptls.nn.AggFeatureSeqEncoder`.
It looks like seq_encoder. It takes raw features at input and provides reduced representation at output.
This encoder creates features, which are good for boosting model. This is a strong baseline for many tasks.
`AggFeatureSeqEncoder` takes the same input as other seq_encoders, and it can easily be replaced
by rnn of transformer seq encoder.  It uses gpu and works fast. It doesn't have parameters for learn.

Possible pipeline:
```python
seq_encoder = AggFeatureSeqEncoder(...)
agg_embeddings = trainer.predict(seq_encoder, dataloader)
catboost_model.fit(agg_embeddings, target)
```

We plan to split `AggFeatureSeqEncoder` into components which will be compatible with other ptls-layers.
It will be possible to choose flexible between `TrxEncoder` with `AggSeqEncoder` and `OheEncoder` with `RnnEncoder`.


## Classes
See docstrings for classes.

Take trx embedding as input:

- `ptls.nn.RnnEncoder`
- `ptls.nn.TransformerEncoder`
- `ptls.nn.LongformerEncoder`

Take raw features as input:

- `ptls.nn.RnnSeqEncoder`
- `ptls.nn.TransformerSeqEncoder`
- `ptls.nn.LongformerSeqEncoder`
- `ptls.nn.AggFeatureSeqEncoder`
