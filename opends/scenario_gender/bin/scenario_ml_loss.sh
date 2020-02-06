# ContrastiveLoss (positive stronger)
export SC_SUFFIX="loss_contrastive_margin_0.5"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=0.5 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# ContrastiveLoss (negative stronger)
export SC_SUFFIX="loss_contrastive_margin_1.0"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=1.0 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# BinomialDevianceLoss (positive stronger)
export SC_SUFFIX="loss_binomialdeviance_C_1.0_alpha_1.0_beta_0.4"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="BinomialDevianceLoss" \
    params.train.C=1.0 \
    params.train.alpha=1.0 \
    params.train.beta=0.4 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# BinomialDevianceLoss (negative stronger)
export SC_SUFFIX="loss_binomialdeviance_C_6.0_alpha_0.4_beta_0.7"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="BinomialDevianceLoss" \
    params.train.C=6.0 \
    params.train.alpha=0.4 \
    params.train.beta=0.7 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json


# TripletLoss
export SC_SUFFIX="loss_triplet_margin_0.3"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="TripletLoss" \
    params.train.margin=0.3 \
    params.train.sampling_strategy="HardTriplets" \
    params.train.neg_count=5 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

export SC_SUFFIX="loss_triplet_margin_0.6"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="TripletLoss" \
    params.train.margin=0.6 \
    params.train.sampling_strategy="HardTriplets" \
    params.train.neg_count=5 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# HistogramLoss
export SC_SUFFIX="loss_histogramloss"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="HistogramLoss" \
    params.train.num_steps=25 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# MarginLoss (positive stronger)
export SC_SUFFIX="loss_margin_0.2_beta_0.4"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="MarginLoss" \
    params.train.margin=0.2 \
    params.train.beta=0.4 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# MarginLoss (negative stronger)
export SC_SUFFIX="loss_margin_0.3_beta_0.6"
python metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss="MarginLoss" \
    params.train.margin=0.3 \
    params.train.beta=0.6 \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json
python ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="../data/gender/emb__$SC_SUFFIX" \
    --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json


# Compare
python -m scenario_gender compare_approaches --output_file "runs/scenario_gender__loss.csv" \
    --skip_baseline --target_score_file_names --ml_embedding_file_names \
    "emb__loss_contrastive_margin_0.5.pickle"                      \
    "emb__loss_contrastive_margin_1.0.pickle"                      \
    "emb__loss_binomialdeviance_C_1.0_alpha_1.0_beta_0.4.pickle"   \
    "emb__loss_binomialdeviance_C_6.0_alpha_0.4_beta_0.7.pickle"   \
    "emb__loss_triplet_margin_0.3.pickle"                          \
    "emb__loss_triplet_margin_0.6.pickle"                          \
    "emb__loss_histogramloss.pickle"                               \
    "emb__loss_margin_0.2_beta_0.4.pickle"                         \
    "emb__loss_margin_0.3_beta_0.6.pickle"
