for SC_HIDDEN_SIZE in 3072 2048 1024 0512 0256 0128 0064 0032
do
  export SC_SUFFIX="hidden_size_${SC_HIDDEN_SIZE}"
  python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
    model_path="models/mles__$SC_SUFFIX.p" \
    data_module.train.batch_size=32 \
    --conf "conf/mles_params.hocon"
  
  python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb_mles__$SC_SUFFIX" \
    inference_dataloader.loader.batch_size=32 \
    --conf "conf/mles_params.hocon"
done

# Compare
rm results/scenario_rosbank__hidden_size.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_rosbank__hidden_size.txt",
      auto_features: ["../data/emb_mles__hidden_size_*.pickle"]'
