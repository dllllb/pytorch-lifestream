for SC_AMOUNT in 3600 1800 0900 0450 0225
do
	python -m scenario_rosbank fit_target \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=20 \
        output.test.path="data/target_scores_$SC_AMOUNT"/test \
        output.valid.path="data/target_scores_$SC_AMOUNT"/valid \
        stats.feature_name="target_scores_${SC_AMOUNT}" \
        stats.path="results/fit_target_${SC_AMOUNT}_results.json" \
        --conf conf/dataset.hocon conf/fit_target_params.json

  python -m scenario_rosbank fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=10 \
        output.test.path="data/mles_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/mles_finetuning_scores_$SC_AMOUNT"/valid \
        stats.feature_name="mles_finetuning_${SC_AMOUNT}" \
        stats.path="results/mles_finetuning_${SC_AMOUNT}_results.json" \
        --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json

  python -m scenario_rosbank fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=10 \
        output.test.path="data/cpc_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/cpc_finetuning_scores_$SC_AMOUNT"/valid \
        stats.feature_name="cpc_finetuning_${SC_AMOUNT}" \
        stats.path="results/cpc_finetuning_${SC_AMOUNT}_results.json" \
        --conf conf/dataset.hocon conf/fit_finetuning_on_cpc_params.json
done

rm results/scenario_rosbank__semi_supervised.txt
# rm -r conf/embeddings_validation_semi_supervised.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_semi_supervised.hocon --workers 10 --total_cpu_count 20
