for SC_AMOUNT in 378 756 1512 3024
do
	python -m scenario_gender fit_target \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=20 \
        output.test.path="../data/gender/target_scores_$SC_AMOUNT"/test \
        output.valid.path="../data/gender/target_scores_$SC_AMOUNT"/valid \
        --conf conf/gender_dataset.hocon conf/gender_target_params_train.json

    python -m scenario_gender fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=10 \
        output.test.path="../data/gender/finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="../data/gender/finetuning_scores_$SC_AMOUNT"/valid \
        --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json

    python -m scenario_gender fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.train.n_epoch=10 \
        params.pretrained_model_path="models/gender_cpc_model.p" \
        output.test.path="../data/gender/finetuning_cpc_scores_$SC_AMOUNT"/test \
        output.valid.path="../data/gender/finetuning_cpc_scores_$SC_AMOUNT"/valid \
        --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json

    # Compare
    python -m scenario_gender compare_approaches \
        --skip_emb_baselines --skip_linear --skip_xgboost \
        --target_score_file_names \
            target_scores_$SC_AMOUNT \
            finetuning_scores_$SC_AMOUNT \
            finetuning_cpc_scores_$SC_AMOUNT \
        --labeled_amount $SC_AMOUNT \
        --output_file runs/semi_scenario_gender_$SC_AMOUNT.csv \
        --ml_embedding_file_names "embeddings.pickle" "embeddings_cpc.pickle"

done
