# CPC
checkpoints_folder="lightning_logs/cpc_model/version_0/checkpoints/*.ckpt"
output_file_prefix="data/cpc__"
conf_file="conf/cpc_params.hocon"
batch_size=1024

for model_file in $(ls -vr $checkpoints_folder)
do
  echo "--------: $model_file"
  model_name=$(basename "$model_file")
  model_file=${model_file//"="/"\="}

  # echo $model_name  # epoch=9-step=44889.ckpt
  epoch_num=$(echo $model_name | cut -f 1 -d "-")
  epoch_num=$(echo $epoch_num | cut -f 2 -d "=")
  epoch_num=$(printf %03d $epoch_num)
  # echo $epoch_num

  output_file=$(echo $output_file_prefix$epoch_num)

  if [ -f "$output_file.pickle" ]; then
    echo "--------: $output_file exists"
  else
    echo "--------: Run inference for $output_file"
    python ../../pl_inference.py model_path="${model_file}" output.path="${output_file}" inference_dataloader.loader.batch_size=${batch_size} --conf "${conf_file}"
  fi
done

rm results/epochs_cpc.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/epochs_cpc.txt",
      auto_features: ["../data/cpc__???.pickle"]'
