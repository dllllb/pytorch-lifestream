for PRJ_SIZE in 256 128 064
do
    for RNN_SIZE in 0128 0256 0512 1024
    do
        export SC_SUFFIX="projection_head_rnn${RNN_SIZE}_prh${PRJ_SIZE}"
        python ../../metric_learning.py \
            params.device="$SC_DEVICE" \
            params.rnn.hidden_size=${RNN_SIZE} \
            params.projection_head.output_size=${PRJ_SIZE} \
            model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
            --conf conf/dataset.hocon conf/mles_proj_head_params.json
        python ../../ml_inference.py \
            params.device="$SC_DEVICE" \
            model_path.model="models/gender_mlm__$SC_SUFFIX.p" \
            output.path="data/emb__$SC_SUFFIX" \
            --conf conf/dataset.hocon conf/mles_proj_head_params.json
    done
done

# Compare
python -m scenario_gender compare_approaches --output_file "results/scenario_gender__projection_head.csv" \
    --models lgb --embedding_file_names \
    "emb__projection_head_rnn0128_prh256.pickle" \
    "emb__projection_head_rnn0256_prh256.pickle" \
    "emb__projection_head_rnn0512_prh256.pickle" \
    "emb__projection_head_rnn1024_prh256.pickle" \
    "emb__projection_head_rnn0128_prh128.pickle" \
    "emb__projection_head_rnn0256_prh128.pickle" \
    "emb__projection_head_rnn0512_prh128.pickle" \
    "emb__projection_head_rnn1024_prh128.pickle" \
    "emb__projection_head_rnn0128_prh064.pickle" \
    "emb__projection_head_rnn0256_prh064.pickle" \
    "emb__projection_head_rnn0512_prh064.pickle" \
    "emb__projection_head_rnn1024_prh064.pickle"

