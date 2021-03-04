
# train special model for fine-tunnig in semi-supervised setup 
# it is quite smaller, than one which is used in supervised setup, due to insufficiency labeled data to train a big model. 
python ../../pl_train_module.py \
    params.rnn.hidden_size=160 \
	model_path="models/age_pred_ml_model_ss_ft.p" \
    --conf "conf/mles_params.hocon"

for SC_AMOUNT in 00337 00675 01350 02700 05400 10800 21600
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
        data_module.train.labeled_amount=$SC_AMOUNT \
        params.rnn.hidden_size=160 \
        params.pretrained_model_path="models/age_pred_ml_model_ss_ft.p" \
        embedding_validation_results.feature_name="mles_finetuning_${SC_AMOUNT}" \
        embedding_validation_results.output_path="results/mles_finetuning_${SC_AMOUNT}_results.json" \
        --conf conf/pl_fit_finetuning_mles.hocon

    python ../../pl_fit_target.py \
        logger_name="cpc_finetuning_${SC_AMOUNT}" \
        data_module.train.labeled_amount=$SC_AMOUNT \
        params.pretrained_model_path="models/cpc_model.p" \
        embedding_validation_results.feature_name="cpc_finetuning_${SC_AMOUNT}" \
        embedding_validation_results.output_path="results/cpc_finetuning_${SC_AMOUNT}_results.json" \
        --conf conf/pl_fit_finetuning_cpc.hocon
done

rm results/scenario_age_pred__semi_supervised.txt
python -m embeddings_validation \
  --conf conf/embeddings_validation_semi_supervised.hocon \
  --workers 10 --total_cpu_count 10