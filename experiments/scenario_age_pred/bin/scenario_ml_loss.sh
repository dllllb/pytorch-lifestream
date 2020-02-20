# ContrastiveLoss (positive stronger)
export SC_SUFFIX="loss_contrastive_margin_0.5"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=0.5 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# ContrastiveLoss (negative stronger)
export SC_SUFFIX="loss_contrastive_margin_1.0"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=1.0 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# BinomialDevianceLoss (positive stronger)
export SC_SUFFIX="loss_binomialdeviance_C_1.0_alpha_1.0_beta_0.4"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="BinomialDevianceLoss" \
    params.train.C=1.0 \
    params.train.alpha=1.0 \
    params.train.beta=0.4 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# BinomialDevianceLoss (negative stronger)
export SC_SUFFIX="loss_binomialdeviance_C_6.0_alpha_0.4_beta_0.7"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="BinomialDevianceLoss" \
    params.train.C=6.0 \
    params.train.alpha=0.4 \
    params.train.beta=0.7 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# TripletLoss
export SC_SUFFIX="loss_triplet_margin_0.3"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="TripletLoss" \
    params.train.margin=0.3 \
    params.train.sampling_strategy="HardTriplets" \
    params.train.neg_count=5 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

export SC_SUFFIX="loss_triplet_margin_0.6"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="TripletLoss" \
    params.train.margin=0.6 \
    params.train.sampling_strategy="HardTriplets" \
    params.train.neg_count=5 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# HistogramLoss
export SC_SUFFIX="loss_histogramloss"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="HistogramLoss" \
    params.train.num_steps=25 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# MarginLoss (positive stronger)
export SC_SUFFIX="loss_margin_0.2_beta_0.4"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="MarginLoss" \
    params.train.margin=0.2 \
    params.train.beta=0.4 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# MarginLoss (negative stronger)
export SC_SUFFIX="loss_margin_0.3_beta_0.6"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="MarginLoss" \
    params.train.margin=0.3 \
    params.train.beta=0.6 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"


# Compare
python -m scenario_age_pred compare_approaches --output_file "results/scenario_age_pred__loss.csv" \
    --embedding_file_names \
    "emb__loss_contrastive_margin_0.5.pickle"                      \
    "emb__loss_contrastive_margin_1.0.pickle"                      \
    "emb__loss_binomialdeviance_C_1.0_alpha_1.0_beta_0.4.pickle"   \
    "emb__loss_binomialdeviance_C_6.0_alpha_0.4_beta_0.7.pickle"   \
    "emb__loss_triplet_margin_0.3.pickle"                          \
    "emb__loss_triplet_margin_0.6.pickle"                          \
    "emb__loss_histogramloss.pickle"                               \
    "emb__loss_margin_0.2_beta_0.4.pickle"                         \
    "emb__loss_margin_0.3_beta_0.6.pickle"
