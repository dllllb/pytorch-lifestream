# HardNegativePair
export SC_SUFFIX="smpl_strategy_HardNegativePair_bs_512_neg_count_5"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="HardNegativePair" \
    params.train.neg_count=5 \
    params.train.batch_size=512 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="smpl_strategy_HardNegativePair_bs_512_neg_count_9"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="HardNegativePair" \
    params.train.neg_count=9 \
    params.train.batch_size=512 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json

export SC_SUFFIX="smpl_strategy_HardNegativePair_bs_512_neg_count_2"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="HardNegativePair" \
    params.train.neg_count=2 \
    params.train.batch_size=512 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json


# AllPositivePair
export SC_SUFFIX="smpl_strategy_AllPositivePair"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="AllPositivePair" \
    params.train.batch_size=512 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json


# DistanceWeightedPair
export SC_SUFFIX="smpl_strategy_DistanceWeightedPair"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="DistanceWeightedPair" \
    params.train.n_samples_from_class=5 \
    params.train.batch_size=512 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="../data/age-pred/emb__$SC_SUFFIX" \
    --conf conf/age_pred_dataset.hocon conf/age_pred_ml_params_inference.json


# Compare
python -m scenario_age_pred compare_approaches --output_file "runs/scenario_age_pred__smpl_strategy.csv" \
    --skip_baseline --target_score_file_names --ml_embedding_file_names \
    "emb__smpl_strategy_HardNegativePair_bs_512_neg_count_5.pickle" \
    "emb__smpl_strategy_HardNegativePair_bs_512_neg_count_9.pickle" \
    "emb__smpl_strategy_HardNegativePair_bs_512_neg_count_2.pickle" \
    "emb__smpl_strategy_AllPositivePair.pickle"                     \
    "emb__smpl_strategy_DistanceWeightedPair.pickle"
