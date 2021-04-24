# Train a supervised model and save scores to the file
python ../../pl_fit_target.py trainer.max_epochs=1 --conf conf/pl_fit_target.hocon

# Fine tune the CPC model in supervised mode and save scores to the file
for i in 20 30 40 50; do
    export split_count=$i
    export SC_SUFFIX="cpc_v2_sub_seq_sampl_strategy_split_count_${split_count}"
    echo "${SC_SUFFIX}"
    python ../../pl_fit_target.py \
        logger_name=${SC_SUFFIX} \
        params.pretrained.model_path="models/$SC_SUFFIX.p" \
        embedding_validation_results.output_path="results/$SC_SUFFIX.json" \
        embedding_validation_results.feature_name="cpc_v2_finetuning_split_count_$split_count" \
        --conf conf/cpc_v2_pl_fit_finetuning.hocon
done


# Compare
rm results/scenario_bowl_baselines_supervised_cpc_v2.txt

python -m embeddings_validation \
    --conf conf/cpc_v2_embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20 --local_scheduler
