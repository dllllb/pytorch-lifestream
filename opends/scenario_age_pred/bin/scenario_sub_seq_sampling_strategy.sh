export SC_SUFFIX="SampleRandom"
export SC_STRATEGY="SampleRandom"
python metric_learning.py \
    params.train.split_strategy.split_strategy=$SC_STRATEGY \
    params.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    params.train.split_strategy.cnt_min=200 \
    params.train.split_strategy.cnt_max=600 \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json


export SC_SUFFIX="SplitRandom"
export SC_STRATEGY="SplitRandom"
python metric_learning.py \
    params.train.split_strategy.split_strategy=$SC_STRATEGY \
    params.valid.split_strategy.split_strategy=$SC_STRATEGY \
    params.train.max_seq_len=600 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json


# Compare
python -m scenario_age_pred compare_approaches --output_file "runs/scenario_age_pred__subseq_smpl_strategy.csv" \
    --skip_baseline --target_score_file_names --ml_embedding_file_names \
    "emb__SplitRandom.pickle"
    "emb__SampleRandom.pickle"


