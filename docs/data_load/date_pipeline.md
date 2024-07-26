# Data pipeline

All processes support `map` and `iterable` data.

There are steps in pipeline:

1. Data preparation:
    - Split folds here
    - Save to parquet file or keep in memory
    - Data format is a dict with feature arrays or scalars
3. Data interface - provide access to any prepared data
    - Data format is a dict with feature arrays or scalars
    - Data can be taken with `__get_item__` of `__iter__` methods
    - Input arrays are any type, output are `torch.Tensor`
    - Tuple samples aren't supported at this stage
    - Parquet file reading are here
    - Augmentations are here
4. Data endpoints - provide a dataloader
    - Target for supervised task extracted here from dict
    - Target for unsupervised task defined here
    - No augmentation here. Should be implemented as part of endpoints if required
