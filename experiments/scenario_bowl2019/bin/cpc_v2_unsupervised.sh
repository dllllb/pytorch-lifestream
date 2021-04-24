# Prepare agg feature encoder and take embedidngs; inference
python ../../pl_train_module.py --conf conf/agg_features_params.hocon
python ../../pl_inference.py --conf conf/agg_features_params.hocon

# Train the Contrastive Predictive Coding (CPC_V2) model; inference
for i in 20 30 40 50; do
    let min_seq_len=$i*5
    export split_count=$i
    export SC_SUFFIX="cpc_v2_sub_seq_sampl_strategy_split_count_${split_count}"
    echo "${SC_SUFFIX}"

    python ../../pl_train_module.py \
        logger_name=${SC_SUFFIX} \
        data_module.train.min_seq_len=$min_seq_len \
        data_module.train.split_strategy.split_count=$split_count \
        \
        data_module.valid.min_seq_len=$min_seq_len \
        data_module.valid.split_strategy.split_count=$split_count \
        model_path="models/$SC_SUFFIX.p" \
        --conf "conf/cpc_v2_params.hocon"

    python ../../pl_inference.py \
        model_path="models/$SC_SUFFIX.p" \
        output.path="data/emb__$SC_SUFFIX" \
        --conf "conf/cpc_v2_params.hocon"
done

rm results/scenario_bowl_baselines_unsupervised_cpc_v2.txt
python -m embeddings_validation \
    --conf conf/cpc_v2_embeddings_validation_baselines_unsupervised.hocon --workers 10 --total_cpu_count 20 --local_scheduler \
    --conf_extra \
      'auto_features: ["../data/emb__cpc_v2_sub_seq_sampl_strategy*.pickle"]'