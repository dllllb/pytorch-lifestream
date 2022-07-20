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
6. Split and groups dataframe by users. One row was one transaction, one row became a user with a list of transactions.
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

```python


```
