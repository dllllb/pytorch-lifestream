# ReduceLROnPlateau
export SC_SUFFIX="lr_reduce_on_plateau"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.lr_scheduler.ReduceLROnPlateau=true \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# ReduceLROnPlateau x2 epochs
export SC_SUFFIX="lr_reduce_on_plateau_x2epochs"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.lr_scheduler.ReduceLROnPlateau=true \
    params.lr_scheduler.threshold=0.0001 \
    params.train.n_epoch=300 \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# CosineAnnealing
export SC_SUFFIX="lr_cosine_annealing"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.lr_scheduler.CosineAnnealing=true \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"
python ../../ml_inference.py \
    params.device="$SC_DEVICE" \
    model_path.model="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf "conf/dataset.hocon" "conf/mles_params.json"

# Compare
rm results/scenario_rosbank_lr_schedule.txt
# rm -r conf/embeddings_validation.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_rosbank_lr_schedule.txt",
      auto_features: ["../data/emb_mles__lr_*.pickle"]'
