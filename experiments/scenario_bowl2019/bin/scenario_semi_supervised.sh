
for SC_AMOUNT in 00496 00994 01986 03971 07943 15887
do
    
    python ../../pl_fit_target.py \
        logger_name="fit_target_${SC_AMOUNT}" \
        params.labeled_amount=$SC_AMOUNT \
        data_module.train.labeled_amount=$SC_AMOUNT \
        embedding_validation_results.feature_name="target_scores_${SC_AMOUNT}" \
        embedding_validation_results.output_path="results/fit_target_${SC_AMOUNT}_results.json" \
        --conf conf/pl_fit_target.hocon
    
    python ../../pl_fit_target.py \
        logger_name="mles_finetuning_${SC_AMOUNT}" \
        data_module.train.labeled_amount=$SC_AMOUNT \
        data_module.train.drop_last=true \
        params.labeled_amount=$SC_AMOUNT \
        params.train.frooze_trx_encoder=true \
        params.train.n_epoch=15 \
        params.train.lr_scheduler.step_gamma=0.5 \
        params.train.lr_scheduler.step_size=5 \
        params.train.lr=0.01 \
        embedding_validation_results.feature_name="mles_finetuning_${SC_AMOUNT}" \
        embedding_validation_results.output_path="results/mles_finetuning_${SC_AMOUNT}_results.json" \
        --conf conf/pl_fit_finetuning_mles.hocon

    python ../../pl_fit_target.py \
        logger_name="cpc_finetuning_${SC_AMOUNT}" \
        data_module.train.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=15 \
        params.train.lr_scheduler.step_gamma=0.1 \
        params.train.lr_scheduler.step_size=10 \
        params.train.lr=0.001 \
        params.pretrained_model_path="models/cpc_model.p" \
        embedding_validation_results.feature_name="cpc_finetuning_${SC_AMOUNT}" \
        embedding_validation_results.output_path="results/cpc_finetuning_${SC_AMOUNT}_results.json" \
        --conf conf/pl_fit_finetuning_cpc.hocon
done

rm results/scenario_bowl2019__semi_supervised.txt
# rm -r conf/embeddings_validation_semi_supervised.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_semi_supervised.hocon --workers 10 --total_cpu_count 20 --local_scheduler
