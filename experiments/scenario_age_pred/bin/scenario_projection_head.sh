for PRJ_SIZE in 256 128 064
do
    for RNN_SIZE in 1600 0800 0400
    do
        export SC_SUFFIX="projection_head_rnn${RNN_SIZE}_prh${PRJ_SIZE}"
        python ../../metric_learning.py \
            params.device="$SC_DEVICE" \
            params.rnn.hidden_size=${RNN_SIZE} \
            params.projection_head.output_size=${PRJ_SIZE} \
            model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
            --conf "conf/dataset.hocon" "conf/mles_proj_head_params.json"
        python ../../ml_inference.py \
            params.device="$SC_DEVICE" \
            model_path.model="models/age_pred_mlm__$SC_SUFFIX.p" \
            output.path="data/emb__$SC_SUFFIX" \
            --conf "conf/dataset.hocon" "conf/mles_proj_head_params.json"
    done
done

# Compare
python -m scenario_age_pred compare_approaches --output_file "results/scenario_age_pred__proj_head.csv" \
    --embedding_file_names \
    "emb__projection_head_rnn1600_prh256.pickle" \
    "emb__projection_head_rnn0800_prh256.pickle" \
    "emb__projection_head_rnn0400_prh256.pickle" \
    "emb__projection_head_rnn1600_prh128.pickle" \
    "emb__projection_head_rnn0800_prh128.pickle" \
    "emb__projection_head_rnn0400_prh128.pickle" \
    "emb__projection_head_rnn1600_prh064.pickle" \
    "emb__projection_head_rnn0800_prh064.pickle" \
    "emb__projection_head_rnn0400_prh064.pickle"
