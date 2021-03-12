for SC_HIDDEN_SIZE in 0400 0200 0100 0064 0032
do
  export SC_SUFFIX="hidden_size_${SC_HIDDEN_SIZE}"
  python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
    params.train.batch_size=64 \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
done

for SC_HIDDEN_SIZE in 0400 0200 0100 0064 0032
do
  export SC_SUFFIX="hidden_size_${SC_HIDDEN_SIZE}"
  python ../../pl_inference.py \
    inference_dataloader.loader.batch_size=64 \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"
done

# Compare
rm results/scenario_bowl2019__hidden_size.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 --local_scheduler \
    --conf_extra \
      'report_file: "../results/scenario_bowl2019__hidden_size.txt",
      auto_features: ["../data/emb__hidden_size_*.pickle"]'
