# `ptls.nn.head`
All classes from `ptls.nn.head` also available in `ptls.nn`

## `ptls.nn.Head`
`Head` is a composition layer. Content is controlled by parameters.

Such scenarios are possible:

- Empty layer. Do nothing. Can replace a default head: `Head()`
- L2 norm for output embedding: `Head(use_norm_encoder=True)`
- Binary classification head: `Head(objective='classification', input_size=m)`
- Multiclass classification head: `Head(objective='classification', input_size=m, num_classes=n)`
- Multilayer binary classification head: `Head(objective='classification', input_size=m, hidden_layers_sizes=[i, j])`
- Regression head: `Head(objective='regression', input_size=m)`

Their combinations are also possible

## Classes
See docstrings for classes.

- `ptls.nn.Head`
