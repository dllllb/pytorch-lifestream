# Embeddings validation project
Estimate your feature vector quality on downstream task

# Concepts
# File loading
TODO: describe common file load features here

## Target file
Tabular file with id field(s) and target columns.
This is a base for quality estimation.

Id can be one ore many columns. dtypes can be string, int, date or datetime.
Id columns are used for join with other filed. Names of columns should be identical across the files,
and you can rename some columns when file loads.

Targets for all data expected as a single list.
Show the name of file if all targets are in a single file.
Show the list of file names if targets are in a many files.

Duplications are checks. Exception will raised if any or them found.
Use `drop_duplicated_ids=True` for dataset auto-repair. First record will ne kept.

## Id file
Contains only ids columns. Used for split data on folds for train, valid and test.
You can use `Target file` as `Id file`. Target columns will bew removed.

## Feature file
Tabular file with ids and feature columns.
You can estimate this specific feature quality for target prediction,
compare it with features from an other file or mix features from many files and compare ensemble.

Id used for join feature and target file. Join is always left: `target` left join `features` on `id`.
Missing values filled according preprocessing strategy.

## Target options
You can specify additional options for each feature file (of set of files).
You can estimate one feature file with different target options.
Just make one more records in `feature` config with the same file path and different name and options.
- `labeled_amount`, this option reduce amount of labeled data in train.
    Set it as integer (record count) or float (rate of total)

## Validation schema
### Cross validation setup
Only one target file is provided. This tool split it on N folds and use each one for validation.
We have N scores provided by N models trained on N-1 folds each.
Result is a mean of N scores with p% confidence interval.

### Train-test setup
Train and valid target files are provided. Train on 1st and mesure score on 2nd file.
Repeat it N times.
We have N scores provided by N models trained on full train file.
Result is a mean of N scores with p% confidence interval.

### Isolated test (optional)
Isolated test target file can be provided.
All models for both cross-validation and train-test setup estimated on isolated test.
N scores from N models provided.
Result is a mean of N scores with p% confidence interval.

### Check id intersection
Ids in train, valid, test shouldn't be intersected. There is a check in `FoldSplitter`.
There are two options:
 - `split.fit_ids is False`: only check. Raise exception when id intersection detected.
 - `split.fit_ids is True`: check and correction. Remove records which a presented in other splits:
    - remove from `valid` records are presented in `test`
    - remove from `train` records are presented in `test`
    - remove from `train` records are presented in `valid`
Check removed record removed counts in `{work_dir}\folds\folds.json`

## Estimator
Many estimators can be used for learn feature and target relations.
We expect sklearn-like fit-predict interface.

## Metrics
Choose the metrics for feature estimate.

There are predefined metrics like predict time or feature count.

## Feature preprocessing
Some features requires preprocessing, you can configure it.
`preprocessing` config requires list if sklearn-like transformers with parameters.
Preprocessing transformers used for all features for all models.

## Feature report
Next releases. We can estimate features and provide some recommendations about preprocessing.

## Final report
For each metrics we provide on part of report.
Each part contains 3 sections for train, validation and optional for test files.
Each section contains results for different feature files and it combinations.
Each result contains mean with confidence intervals of N scores from N folds (or iterations).
Only metrics presented in `report.metrics` will be in report if `report.print_all_metrics=false`.
All available metrics will be in report if `report.print_all_metrics=true`.

## Report format
Each cell of final report have scores from N iteration.
These scores transformed into report fields:
 - mean
 - t_pm: delta for confidence interval. Real mean is in interval (mean - t_pm, mean + t_pm) with 95% confidence
 - t_int_l, t_int_h: lower and upper borders of confidence interval. Real mean is in interval (t_int_l, t_int_h) with 95% confidence
 - std
 - values: list of all N values
 
You can choose fields which will be presented in reports and float_format of these values.

## External scores
You can estimate model quality yourself and show results in common report.
Show the path where results are presented. This should be json file with such structure:
```json
[
  {
      "fold_id": "<fold_num in range 0..N>",
      "model_name": "<model_name>",
      "feature_name": "<feature_name>",
      "scores_valid": {
        "<results are here, column list depends on your task>": "<values>",
        "auroc": 0.987654,
        "auroc_score_time": 0,
        "accuracy": 0.123456,
        "accuracy_score_time": 0,
        "cnt_samples": 10,
        "cnt_features": 4
      },
      "scores_train": {
          "<the same keys>": "<values>"
      },
      "scores_test": {
          "<the same keys>": "<values>"
      }
  }
]
```
These results will be collected and presented in report.

## Folds for external scores
Your external scorer can use train-valid-test split from embedding_validation.
1. Set `split.row_order_shuffle_seed` for specify random row order in folds.
2. Run `python -m embeddings_validation --split_only`
3. Run your scorer and read fold information:
    ```python
    from embeddings_validation.file_reader import TargetFile
 
    fold_info = json.load(f'{environment.work_dir}/folds/folds.json')
    current_fold = fold_info[fold_id]
   
    train_targets = TargetFile.load(current_fold['train']['path'])
    valid_targets = TargetFile.load(current_fold['valid']['path'])
    test_targets = TargetFile.load(current_fold['test']['path'])

    if labeled_amount:
        data = train_targets.df.iloc[:labeled_amount]
    else:
        data = train_targets.df
    ```
4. Save scores as was described before


## Compare with baseline
You can compare result of our model with baseline.
Define which row in your report is baseline (specify model name and feature name).
Independent Two-sample t Test is used.
1st samples are baseline scores (X). 2nd samples are scores from one specific report row (Y).
Equal variances assumed due to data origin. And F-test of equality of variances are used.
Next values will be added in report:
- t_f_stat: statistic of F-test
- t_f_alpha: probability to have such or less t_f_stat value
- t_t_stat: statistic of T-test (positive value show that row scores are greater than baseline)
- t_t_alpha: probability to have such or less t_t_stat value
- t_A (removed): gap between X and Y which can be reduced with keeping statistically significant difference between X and Y
- t_A_pp (removed): t_A in percent above baseline
- delta: absolute difference between row scores and baseline
- delta_pm: confidence interval delta for previous metric
- delta_l: delta - delta_pm, lower bound of confidence interval
- delta_h: delta + delta_pm, upper bound of confidence interval
- delta_pp: delta in percent above baseline
- delta_pm_pp: delta_pm in percent above baseline
- delta_l_pp: delta_l in percent above baseline
- delta_h_pp: delta_h in percent above baseline


## Error handling
Some folds can fail during execution. There are some strategies to handle it:
  - fail: don't collect report until all folds will be correct
  - skip: collect report without missing scores. Allow rerun failed task

## Config
All settings should be described in single configuration file.
Report preparation splits on task and execution plan prepared based on config.


# Execution plan
1. Read target files, split it based on validation schema.
2. (Next release) Run feature check task for each feature file
3. Based on feature file combinations prepare feature estimate tasks.
2. Run train-estimate task for each split. Save results.
3. Collect report


# Environment
`embeddings_validation` module should be available for current python.

Directory with config file is `root_path`. All paths described in config starts from `root_path`.
`config.work_dir` is a directory where intermediate files and reports are saved.
`config.report_file` is a file with final report.

# Resources
Required cpu count should be defined with model.
Total cpu count limit and num_workers should be defined when application run.
Estimation will works in parallel with cpu limit.

# Git bundle distribution
```
# build server
git bundle create embeddings_validation.bundle HEAD master

# transfer file embeddings_validation.bundle to an other server
# via email

# client server 1st time
git clone embeddings_validation.bundle embeddings_validation
# client server next time
git pull

```

# Install package
```
python3 setup.py install

```

# Build package (not used)
Taken from [here](https://packaging.python.org/tutorials/packaging-projects/).

```
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel

# check dist directory

```

# How to run
```
# run server first
luigid

# for local run without central server add `--local_scheduler` option to `python -m embeddings_validation` args

```

## Test example `train-test.hocon`
```
# delete old files
rm -r test_conf/train-test.work/
rm test_conf/train-test.txt

# run report collection
python -m embeddings_validation --workers 4 --conf test_conf/train-test.hocon --total_cpu_count 10

# check final report
less test_conf/train-test.txt
```
## Test example `train-valid-1iter.hocon`
```
# delete old files
rm -r test_conf/train-valid-1iter.work/
rm test_conf/train-valid-1iter.txt

# run report collection
python -m embeddings_validation --workers 4 --conf test_conf/train-valid-1iter.hocon --total_cpu_count 10

# check final report
less test_conf/train-valid-1iter.txt
```
## Test example `crossval.hocon`
```
# delete old files
rm -r test_conf/crossval.work/
rm test_conf/crossval.txt

# run report collection
python -m embeddings_validation --workers 4 --conf test_conf/crossval.hocon --total_cpu_count 10

# check final report
less test_conf/crossval.txt
```

## Test example `single-file.hocon`
```
# delete old files
rm -r test_conf/single-file.work/
rm test_conf/single-file.txt

# run report collection
python -m embeddings_validation --workers 4 --conf test_conf/single-file.hocon --total_cpu_count 10

# check final report
less test_conf/single-file.txt
```

## Test example `single-file-short.hocon`
```
# delete old files
rm -r test_conf/single-file-short.work/
rm test_conf/single-file-short.txt

# run report collection
python -m embeddings_validation --workers 4 --conf test_conf/single-file-short.hocon --total_cpu_count 10

# check final report
less test_conf/single-file-short.txt
```

# TODO
- report config summary, print warnings for smells options
- features check:
    - is category
    - is datetime
    - is target col in feature (by name or by content similarity)
