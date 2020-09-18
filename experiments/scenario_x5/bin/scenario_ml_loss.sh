# ContrastiveLoss (positive stronger)
export SC_SUFFIX="loss_contrastive_margin_0.5"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=0.5 \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# ContrastiveLoss (negative stronger)
export SC_SUFFIX="loss_contrastive_margin_1.0"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=1.0 \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# BinomialDevianceLoss (positive stronger)
export SC_SUFFIX="loss_binomialdeviance"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="BinomialDevianceLoss" \
    params.train.C=1.0 \
    params.train.alpha=1.0 \
    params.train.beta=0.3 \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# TripletLoss
export SC_SUFFIX="loss_triplet"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="TripletLoss" \
    params.train.margin=0.6 \
    params.train.sampling_strategy="HardTriplets" \
    params.train.neg_count=5 \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# HistogramLoss
export SC_SUFFIX="loss_histogramloss"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="HistogramLoss" \
    params.train.num_steps=51 \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# MarginLoss (positive stronger)
export SC_SUFFIX="loss_margin_0.2_beta_0.4"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="MarginLoss" \
    params.train.margin=0.2 \
    params.train.beta=0.4 \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# MarginLoss (negative stronger)
export SC_SUFFIX="loss_margin_0.3_beta_0.6"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="MarginLoss" \
    params.train.margin=0.3 \
    params.train.beta=0.6 \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# Compare
python -m scenario_x5 compare_approaches --output_file "results/scenario_x5__loss.csv" \
    --n_workers 2 --models lgb --embedding_file_names \
    "mles_embeddings.pickle"                    \
    "emb__loss_*.pickle"
