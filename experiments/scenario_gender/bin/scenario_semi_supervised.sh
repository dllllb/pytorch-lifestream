for SC_AMOUNT in 378 756 1512 3024 6048
do
	python -m scenario_gender fit_target \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=20 \
        output.test.path="data/target_scores_$SC_AMOUNT"/test \
        output.valid.path="data/target_scores_$SC_AMOUNT"/valid \
        --conf conf/dataset.hocon conf/fit_target_params.json

    python -m scenario_gender fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=10 \
        output.test.path="data/finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/finetuning_scores_$SC_AMOUNT"/valid \
        --conf conf/dataset.hocon conf/mles_finetuning_params.json

    python -m scenario_gender fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=10 \
        output.test.path="data/cpc_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/cpc_finetuning_scores_$SC_AMOUNT"/valid \
        --conf conf/dataset.hocon conf/cpc_finetuning_params.json

    # Compare
    python -m scenario_gender compare_approaches \
        --models "lgb" \
        --score_file_names \
            target_scores_$SC_AMOUNT \
            finetuning_scores_$SC_AMOUNT \
            finetuning_cpc_scores_$SC_AMOUNT \
        --labeled_amount $SC_AMOUNT \
        --output_file results/semi_scenario_gender_$SC_AMOUNT.csv \
        --embedding_file_names "embeddings.pickle" "embeddings_cpc.pickle"

done
