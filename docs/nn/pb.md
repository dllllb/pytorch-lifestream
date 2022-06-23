# `ptls.nn.pb`
All classes from `ptls.nn.pb` also available in `ptls.nn`

All classes in this module support `PaddedBatch` as input and output.
Many modules extend `torch.nn` classes.

## Inherited layers
Some layers are inherited from the original classes with forward reimplement. Original forward process `x.payload`.
Result are packed to `PaddedBatch`. `x.seq_lens` passed to output `PaddedBatch`.

PB-layers keep original class behavioral.

Example:
```python
x = PaddedBatch(torch.randn(4, 12, 8), torch.LongTensor([3, 12, 8]))
model = PBLinear(8, 5)
y = model(x)
assert y.payload.size() == (4, 12, 5)
help(PBLinear)
```

PB-layers can be used with other layers in `torch.nn.Sequential`

```python
x = PaddedBatch(torch.randn(4, 12, 8), torch.LongTensor([3, 12, 8]))
model = torch.nn.Sequential(
    PBLinear(8, 5),
    PBReLU(),
    PBLinear(5, 10),
)
y = model(x)
assert y.payload.size() == (4, 12, 10)
```

### Class mapping

| Pb layer    | Parent Layer           |
| ----------- | ---------------------- | 
| PBLinear    | torch.nn.Linear        |
| PBLayerNorm | torch.nn.LayerNorm     |
| PBReLU      | torch.nn.ReLU          |
| PBL2Norm    | ptls.nn.L2NormEncoder  |

## Classes
See docstrings for classes.

- `ptls.nn.PBLinear`
- `ptls.nn.PBLayerNorm`
- `ptls.nn.PBL2Norm`
- `ptls.nn.PBReLU`
- `ptls.nn.PBL2Norm`
