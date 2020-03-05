for RNN_SIZE in 1600 0800 0400
do
    export SC_SUFFIX="projection_head_rnn${RNN_SIZE}_prh256"
    python ../../metric_learning.py \
        params.device="$SC_DEVICE" \
        params.rnn.hidden_size=${RNN_SIZE} \
        params.projection_head.output_size=256 \
        model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
        --conf "conf/dataset.hocon" "conf/mles_params.json"
    python ../../ml_inference.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
        output.path="data/emb__$SC_SUFFIX" \
        --conf "conf/dataset.hocon" "conf/mles_params.json"
done

for RNN_SIZE in 1600 0800 0400
do
    export SC_SUFFIX="projection_head_rnn${RNN_SIZE}_prh128"
    python ../../metric_learning.py \
        params.device="$SC_DEVICE" \
        params.rnn.hidden_size=${RNN_SIZE} \
        params.projection_head.output_size=128 \
        model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
        --conf "conf/dataset.hocon" "conf/mles_params.json"
    python ../../ml_inference.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
        output.path="data/emb__$SC_SUFFIX" \
        --conf "conf/dataset.hocon" "conf/mles_params.json"
done

# Compare
python -m scenario_age_pred compare_approaches --output_file "results/scenario_age_pred__hidden_size.csv" \
    --embedding_file_names \
    "emb__projection_head_rnn1600_prh256.pickle" \
    "emb__projection_head_rnn0800_prh256.pickle" \
    "emb__projection_head_rnn0400_prh256.pickle" \
    "emb__projection_head_rnn1600_prh128.pickle" \
    "emb__projection_head_rnn0800_prh128.pickle" \
    "emb__projection_head_rnn0400_prh128.pickle"

