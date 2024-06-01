# Preprocessing

Source data usually has different formats. 
`ptls.preprocessing` has a tools to transform it to `ptls`-compatible format.

## `pandas` or `pyspark`

Use `pandas` for a small dataset and `pyspark` for a large one. 
`pyspark` may be in local or cluster mode.

## Steps of preprocessing

1. Load flat transactional data into `pandas.DataFrame` or `spark.DataFrame`
2. Identify user_id column. There are usually no modifications for this column.
3. Prepare `event_time` column. Convert it to a timestamp for a date and time, or use any sortable format otherwise.
4. Fit and transform categorical features, from categorical values to embedding indexes.
5. Check numeric feature column types
6. Split and group dataframe by users. Before this operation a row represents a single transaction; after a row represents a user with and contains a list of all transactions by the user.
7. Join user-level columns: target, labels, features.
8. Done. Use data from memory or save it to parquet format.
9. Save fitted preprocessing for future usage.

These steps are implemented in preprocessor classes: 
`ptls.preprocessing.PandasDataPreprocessor`, `ptls.preprocessing.PysparkDataPreprocessor`.

> **Note**
> 
> We recommend using the minimum set of transformations. This allows more flexible pipeline in the future.

Tips for transformation:

- keep only `event_time` timestamp, don't keep datetime features. This saves a storage and improve load speed.
- keep raw values for numerical features. You can try a variety of normalizing and outlier clip options.
- missing values imputing for categorical features should be done before transformation.
Missing values will be processed as separate embedding.
- missing values imputing for numerical features [TBD]

## Data preprocessors

`ptls.preprocessing.PandasDataPreprocessor`, `ptls.preprocessing.PysparkDataPreprocessor` have a similar interface.
Let's inspect one or them.

Prepare test data:

```python
import numpy as np
import pandas as pd

N_USERS = 200
SEQ_LEN = 20

df_trx = pd.DataFrame({
    'user_id': np.repeat(np.arange(N_USERS), SEQ_LEN),
    'dt': np.datetime64('2016-01-01 00:00:00') + 
    (np.random.rand(N_USERS * SEQ_LEN) * 365 * 24 * 60 * 60).astype('timedelta64'),
    'mcc_code': (np.random.randint(10, 99, N_USERS * SEQ_LEN) * 100).astype(str),
    'amount': np.exp(np.random.randn(N_USERS * SEQ_LEN) + np.log(5000)).round(2)
})

df_trx.head(6)
```

This is dataframe with 200 unique users with 20 transaction in each. Random date, mcc code and amount.

| user_id |                  dt | mcc_code |  amount |
| ------- | ------------------- | -------- | ------- |
|       0 | 2016-01-31 06:35:08 |     7700 | 3366.67 |
|       0 | 2016-05-29 22:42:54 |     8600 | 3513.50 |
|       0 | 2016-06-20 06:14:16 |     7300 | 2738.51 |
|       0 | 2016-10-09 03:10:34 |     6800 |  726.59 |
|       0 | 2016-07-04 06:50:37 |     6200 | 6264.04 |
|       0 | 2016-02-02 17:26:14 |     6100 | 4806.28 |

Let's use a preprocessor

```python
from ptls.preprocessing import PandasDataPreprocessor

preprocessor = PandasDataPreprocessor(
    col_id='user_id',
    col_event_time='dt',
    event_time_transformation='dt_to_timestamp',
    cols_category=['mcc_code'],
    cols_numerical=['amount'],
)

data = preprocessor.fit_transform(df_trx)

data[:2]
```

Output will be like:

```
[{'user_id': 0,
  'mcc_code': tensor([55, 77, 59, 60, 21, 44, 85, 79, 34, 28, 24, 46, 54,  9, 25,  7, 84, 28,
          39, 11]),
  'amount': tensor([ 3692.1400,  3366.6700,  4806.2800,  4048.3000,  3513.5000,  1319.7900,
           2738.5100,  1838.5500,  6264.0400,   676.3800,  2747.5900,  1223.0100,
           1403.7600, 21391.0100,   726.5900,   765.0500,  7832.1700,  2234.4300,
          18762.4900,  3644.8800], dtype=torch.float64),
  'event_time': tensor([1452449213, 1454222108, 1454433974, 1460899926, 1464561774, 1465547819,
          1466403256, 1467196958, 1467615037, 1468001211, 1468322417, 1468575287,
          1469942976, 1471525888, 1475982634, 1478011070, 1479214698, 1479350032,
          1479884254, 1482953189])},
 {'user_id': 1,
  'mcc_code': tensor([ 2, 87, 18, 33, 12, 10, 39, 76, 56, 15, 38, 14, 88, 56, 20, 15, 63, 63,
          19, 11]),
  'amount': tensor([ 6045.3100,  3814.1900,  1808.9300,  7235.7800,  1240.0300,  7085.0500,
          11645.6500,  1935.9500,  4777.8000, 41611.2300,  6154.5100,  4797.5500,
          26597.2400,  5005.9900, 12201.0700, 10061.3800,  3780.7400,  2559.4200,
           7252.6700, 30190.5500], dtype=torch.float64),
  'event_time': tensor([1452063037, 1452609464, 1454020103, 1458081768, 1458243803, 1459655589,
          1460157815, 1461727087, 1463158828, 1463651732, 1464883496, 1466071129,
          1472361876, 1474923172, 1475222978, 1476328691, 1477681257, 1478186343,
          1478460764, 1481779245])}]
```

Let's check:

1. All transactions are split between users
```python
assert len(data) == N_USERS
assert sum(len(rec['event_time']) for rec in data) == len(df_trx)
```

2. `user_id` is a scalar field in dictionary.
3. `event_time` is a timestamp. Sequences are ordered.
4. Categorical features `mcc_code` are encoded to embedding indexes.
5. Numeric feature `amount` is identical as input.
6. Each feature is a tensor. 

The same way is used for `ptls.preprocessing.PysparkDataPreprocessor`.
