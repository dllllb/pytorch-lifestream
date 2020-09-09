# HardNegativePair
export SC_SUFFIX="smpl_strategy_HardNegativePair_neg_count_5"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="HardNegativePair" \
    params.train.neg_count=5 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="smpl_strategy_HardNegativePair_neg_count_9"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="HardNegativePair" \
    params.train.neg_count=9 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="smpl_strategy_HardNegativePair_neg_count_2"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="HardNegativePair" \
    params.train.neg_count=2 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# AllPositivePair
export SC_SUFFIX="smpl_strategy_AllPositivePair"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="AllPositivePair" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# DistanceWeightedPair
export SC_SUFFIX="smpl_strategy_DistanceWeightedPair"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.sampling_strategy="DistanceWeightedPair" \
    params.train.n_samples_from_class=5 \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/trx_dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# Compare
python -m scenario_bowl2019 compare_approaches --output_file "results/scenario_bowl2019__smpl_strategy.csv" \
    --models 'lgb' --embedding_file_names \
    "mles_embeddings.pickle"              \
    "emb__smpl_strategy_HardNegativePair_neg_count_5.pickle" \
    "emb__smpl_strategy_HardNegativePair_neg_count_9.pickle" \
    "emb__smpl_strategy_HardNegativePair_neg_count_2.pickle" \
    "emb__smpl_strategy_AllPositivePair.pickle"                     \
    "emb__smpl_strategy_DistanceWeightedPair.pickle"
