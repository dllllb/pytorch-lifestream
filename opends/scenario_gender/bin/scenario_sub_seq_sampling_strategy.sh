export SC_SUFFIX="SampleRandom"
export SC_STRATEGY="SampleRandom"
python metric_learning.py \
    params.train.split_strategy.split_strategy=$SC_STRATEGY \
    params.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json


export SC_SUFFIX="SplitRandom"
export SC_STRATEGY="SplitRandom"
python metric_learning.py \
    params.train.split_strategy.split_strategy=$SC_STRATEGY \
    params.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json


# Compare
python -m scenario_gender compare_approaches --output_file "runs/scenario_gender__subseq_smpl_strategy.csv" \
    --skip_baseline --target_score_file_names --ml_embedding_file_names \
    "emb__SplitRandom" \
    "emb__SampleRandom" \

