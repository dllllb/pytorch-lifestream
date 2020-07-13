# ReduceLROnPlateau
export SC_SUFFIX="reduce_on_plateau"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.lr_scheduler.ReduceLROnPlateau=true \
    params.lr_scheduler.patience=5 \
    params.lr_scheduler.factor=0.5 \
    params.lr_scheduler.threshold=0.001 \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# CosineAnnealing
export SC_SUFFIX="cosine_annealing"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.lr_scheduler.CosineAnnealing=true \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Compare
python -m scenario_age_pred compare_approaches --output_file "results/scenario_lr_schedule.csv" \
    --n_workers 4 --models lgb --embedding_file_names \
    "mles_embeddings.pickle"        \
    "emb__reduce_on_plateau.pickle" \
    "emb__cosine_annealing.pickle"
