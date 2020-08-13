for SC_AMOUNT in 0378 0756 1512 3024 6048
do
	python -m scenario_gender fit_target \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/target_scores_$SC_AMOUNT"/test \
        output.valid.path="data/target_scores_$SC_AMOUNT"/valid \
        stats.path="results/fit_target_results_$SC_AMOUNT.json" \
        --conf conf/dataset.hocon conf/fit_target_params.json

    python -m scenario_gender fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/finetuning_scores_$SC_AMOUNT"/valid \
        stats.path="results/mles_finetuning_results_$SC_AMOUNT.json" \
        --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json

    python -m scenario_gender fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/cpc_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/cpc_finetuning_scores_$SC_AMOUNT"/valid \
        stats.path="results/cpc_finetuning_results_$SC_AMOUNT.json" \
        --conf conf/dataset.hocon conf/fit_finetuning_on_cpc_params.json

    # Compare
    python -m scenario_gender compare_approaches \
        --models "lgb" \
        --score_file_names \
            target_scores_$SC_AMOUNT \
            finetuning_scores_$SC_AMOUNT \
            cpc_finetuning_scores_$SC_AMOUNT \
        --labeled_amount $SC_AMOUNT \
        --output_file results/semi_scenario_gender_$SC_AMOUNT.csv \
        --baseline_name "agg_feat_embed.pickle" \
        --embedding_file_names "mles_embeddings.pickle" "cpc_embeddings.pickle"
done
