for PRJ_SIZE in 256 128 064
do
    for RNN_SIZE in 0128 0256 0512 1024
    do
        export SC_SUFFIX="projection_head_rnn${RNN_SIZE}_prh${PRJ_SIZE}"
        python ../../pl_train_module.py \
            logger_name=${SC_SUFFIX} \
            params.rnn.hidden_size=${RNN_SIZE} \
            "params.head_layers=[[Linear, {in_features: ${RNN_SIZE}, out_features: ${PRJ_SIZE}}], [BatchNorm1d, {num_features: ${PRJ_SIZE}}], [ReLU, {}], [Linear, {in_features: ${PRJ_SIZE}, out_features: ${PRJ_SIZE}}], [NormEncoder, {}]]" \
            model_path="models/gender_mlm__$SC_SUFFIX.p" \
            --conf conf/mles_proj_head_params.hocon
        python ../../pl_inference.py \
            model_path="models/gender_mlm__$SC_SUFFIX.p" \
            output.path="data/emb__$SC_SUFFIX" \
            --conf conf/mles_proj_head_params.hocon
    done
done

# Compare
rm results/scenario_gender__projection_head.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_gender__projection_head.txt",
      auto_features: ["../data/emb__projection_head_*.pickle"]'
