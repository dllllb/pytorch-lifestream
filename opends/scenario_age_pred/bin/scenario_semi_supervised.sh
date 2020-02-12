for SC_AMOUNT in 337 675 1350 2700 5400 10800 21600
do
	python -m scenario_age_pred fit_target \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="../data/age-pred/target_scores_$SC_AMOUNT"/test \
        output.valid.path="../data/age-pred/target_scores_$SC_AMOUNT"/valid \
        --conf conf/age_pred_dataset.hocon conf/age_pred_target_params_train.json

    python -m scenario_age_pred fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.rnn.hidden_size=160 \
        output.test.path="../data/age-pred/finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="../data/age-pred/finetuning_scores_$SC_AMOUNT"/valid \
        --conf conf/age_pred_dataset.hocon conf/age_pred_finetuning_params_train.json

    python -m scenario_age_pred fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.rnn.hidden_size=160 \
        params.pretrained_model_path="models/age_pred_cpc_model.p" \
        output.test.path="../data/age-pred/finetuning_cpc_scores_$SC_AMOUNT"/test \
        output.valid.path="../data/age-pred/finetuning_cpc_scores_$SC_AMOUNT"/valid \
        --conf conf/age_pred_dataset.hocon conf/age_pred_finetuning_params_train.json

    python -m scenario_age_pred pseudo_labeling \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="../data/age-pred/pseudo_labeling_$SC_AMOUNT"/test \
        output.valid.path="../data/age-pred/pseudo_labeling_$SC_AMOUNT"/valid \
        --conf conf/age_pred_dataset.hocon conf/age_pred_pseudolabel_params_train.json

    # Compare
    python -m scenario_age_pred compare_approaches \
        --skip_emb_baselines --skip_linear --skip_xgboost 
        --target_score_file_names \
            target_scores_$SC_AMOUNT \
            finetuning_scores_$SC_AMOUNT \
            finetuning_cpc_scores_$SC_AMOUNT \
            pseudo_labeling_$SC_AMOUNT \
        --labeled_amount $SC_AMOUNT \
        --output_file runs/semi_scenario_age_pred_$SC_AMOUNT.csv \
        --ml_embedding_file_names "embeddings.pickle" "embeddings_cpc.pickle"
done


