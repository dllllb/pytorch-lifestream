
# train special model for fine-tunnig in semi-supervised setup 
# it is quite smaller, than one which is used in supervised setup, due to insufficiency labeled data to train a big model. 
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.rnn.hidden_size=160 \
	model_path.model="models/age_pred_ml_model_ss_ft.p" \
	--conf "conf/dataset.hocon" "conf/mles_params.json"

for SC_AMOUNT in 00337 00675 01350 02700 05400 10800 21600
do
	python -m scenario_age_pred fit_target \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        output.test.path="data/target_scores_$SC_AMOUNT"/test \
        output.valid.path="data/target_scores_$SC_AMOUNT"/valid \
        stats.feature_name="target_scores_${SC_AMOUNT}" \
        stats.path="results/fit_target_${SC_AMOUNT}_results.json" \
        --conf "conf/dataset.hocon" conf/fit_target_params.json

    python -m scenario_age_pred fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.rnn.hidden_size=160 \
        params.pretrained_model_path="models/age_pred_ml_model_ss_ft.p" \
        output.test.path="data/mles_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/mles_finetuning_scores_$SC_AMOUNT"/valid \
        stats.feature_name="mles_finetuning_${SC_AMOUNT}" \
        stats.path="results/mles_finetuning_${SC_AMOUNT}_results.json" \
        --conf "conf/dataset.hocon" conf/fit_finetuning_on_mles_params.json

    python -m scenario_age_pred fit_finetuning \
        params.device="$SC_DEVICE" \
        params.labeled_amount=$SC_AMOUNT \
        params.pretrained_model_path="models/cpc_model.p" \
        output.test.path="data/cpc_finetuning_scores_$SC_AMOUNT"/test \
        output.valid.path="data/cpc_finetuning_scores_$SC_AMOUNT"/valid \
        stats.feature_name="cpc_finetuning_${SC_AMOUNT}" \
        stats.path="results/cpc_finetuning_${SC_AMOUNT}_results.json" \
        --conf "conf/dataset.hocon" conf/fit_finetuning_on_cpc_params.json

#    python -m scenario_age_pred pseudo_labeling \
#        params.device="$SC_DEVICE" \
#        params.labeled_amount=$SC_AMOUNT \
#        output.test.path="data/pseudo_labeling_$SC_AMOUNT"/test \
#        output.valid.path="data/pseudo_labeling_$SC_AMOUNT"/valid \
#        --conf "conf/dataset.hocon" conf/pseudolabel_params.json

done

rm results/scenario_age_pred__semi_supervised.txt
# rm -r conf/embeddings_validation_semi_supervised.work/
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_semi_supervised.hocon --workers 10 --total_cpu_count 20
