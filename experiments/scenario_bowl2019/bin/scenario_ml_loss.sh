# ContrastiveLoss (positive stronger)
export SC_SUFFIX="loss_contrastive_margin_0.5"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=0.5 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# ContrastiveLoss (negative stronger)
export SC_SUFFIX="loss_contrastive_margin_1.0"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=1.0 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# BinomialDevianceLoss (positive stronger)
export SC_SUFFIX="loss_binomialdeviance_C_1.0_alpha_1.0_beta_0.4"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.loss="BinomialDevianceLoss" \
    params.train.C=1.0 \
    params.train.alpha=1.0 \
    params.train.beta=0.4 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# BinomialDevianceLoss (negative stronger)
export SC_SUFFIX="loss_binomialdeviance_C_6.0_alpha_0.4_beta_0.7"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.loss="BinomialDevianceLoss" \
    params.train.C=6.0 \
    params.train.alpha=0.4 \
    params.train.beta=0.7 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# TripletLoss
export SC_SUFFIX="loss_triplet_margin_0.3"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.loss="TripletLoss" \
    params.train.margin=0.3 \
    params.train.sampling_strategy="HardTriplets" \
    params.train.neg_count=5 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

export SC_SUFFIX="loss_triplet_margin_0.6"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.loss="TripletLoss" \
    params.train.margin=0.6 \
    params.train.sampling_strategy="HardTriplets" \
    params.train.neg_count=5 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# HistogramLoss
export SC_SUFFIX="loss_histogramloss"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.loss="HistogramLoss" \
    params.train.num_steps=25 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# MarginLoss (positive stronger)
export SC_SUFFIX="loss_margin_0.2_beta_0.4"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.loss="MarginLoss" \
    params.train.margin=0.2 \
    params.train.beta=0.4 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# MarginLoss (negative stronger)
export SC_SUFFIX="loss_margin_0.3_beta_0.6"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.loss="MarginLoss" \
    params.train.margin=0.3 \
    params.train.beta=0.6 \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# Compare
rm results/scenario_bowl2019__loss.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 --local_scheduler \
    --conf_extra \
      'report_file: "../results/scenario_bowl2019__loss.txt",
      auto_features: ["../data/emb__loss_*.pickle"]'
