for SC_AMOUNT in 3600 1800 0900 0450 0225
do
      python ../../pl_fit_target.py \
            logger_name="fit_target_${SC_AMOUNT}" \
            trainer.max_epochs=20 \
            data_module.train.drop_last=true \
            data_module.train.labeled_amount=$SC_AMOUNT \
            embedding_validation_results.feature_name="target_scores_${SC_AMOUNT}" \
            embedding_validation_results.output_path="results/fit_target_${SC_AMOUNT}_results.json" \
            --conf conf/pl_fit_target.hocon

      python ../../pl_fit_target.py \
            logger_name="mles_finetuning_${SC_AMOUNT}" \
            trainer.max_epochs=10 \
            data_module.train.drop_last=true \
            data_module.train.labeled_amount=$SC_AMOUNT \
            embedding_validation_results.feature_name="mles_finetuning_${SC_AMOUNT}" \
            embedding_validation_results.output_path="results/mles_finetuning_${SC_AMOUNT}_results.json" \
            --conf conf/pl_fit_finetuning_mles.hocon

      python ../../pl_fit_target.py \
            logger_name="cpc_finetuning_${SC_AMOUNT}" \
            trainer.max_epochs=10 \
            data_module.train.drop_last=true \
            data_module.train.labeled_amount=$SC_AMOUNT \
            embedding_validation_results.feature_name="cpc_finetuning_${SC_AMOUNT}" \
            embedding_validation_results.output_path="results/cpc_finetuning_${SC_AMOUNT}_results.json" \
            --conf conf/pl_fit_finetuning_cpc.hocon
done

rm results/scenario_rosbank__semi_supervised.txt
python -m embeddings_validation \
  --conf conf/embeddings_validation_semi_supervised.hocon \
  --workers 10 --total_cpu_count 18
