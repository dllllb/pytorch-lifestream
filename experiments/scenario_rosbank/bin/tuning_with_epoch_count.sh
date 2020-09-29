#rm -r models/mles_ckp_tuning_*
#rm -r data/mles_emb_tuning_*
#rm -r conf/embeddings_validation.work/
#rm results/scenario_tuning.txt

export SC_SUFFIX="base"
rm -r models/mles_ckp_${SC_SUFFIX}
python ../../metric_learning.py \
  params.device="$SC_DEVICE" \
  params.train.checkpoints.save_interval=10 \
  params.train.checkpoints.n_saved=1000 \
  params.train.checkpoints.dirname="models/mles_ckp_tuning_${SC_SUFFIX}/" \
  params.train.checkpoints.filename_prefix="mles" \
  params.train.checkpoints.create_dir=true \
  --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference_lazy.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/mles_ckp_tuning_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_emb_tuning_${SC_SUFFIX}_${SC_EPOCH}" \
        --conf conf/dataset.hocon conf/mles_params.json
done
rm results/scenario_tuning.txt
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 18 \
    --conf_extra 'report_file: "../results/scenario_tuning.txt", auto_features: ["../data/mles_emb_tuning_*.pickle"]'
less -S results/scenario_tuning.txt


export SC_SUFFIX="lstm"
rm -r models/mles_ckp_tuning_${SC_SUFFIX}
python ../../metric_learning.py \
  params.device="$SC_DEVICE" \
  params.rnn.type="lstm" \
  params.train.n_epoch=300 \
  params.train.checkpoints.save_interval=10 \
  params.train.checkpoints.n_saved=1000 \
  params.train.checkpoints.dirname="models/mles_ckp_tuning_${SC_SUFFIX}/" \
  params.train.checkpoints.filename_prefix="mles" \
  params.train.checkpoints.create_dir=true \
  --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300
do
    python ../../ml_inference_lazy.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/mles_ckp_tuning_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_emb_tuning_${SC_SUFFIX}_${SC_EPOCH}" \
        --conf conf/dataset.hocon conf/mles_params.json
done
rm results/scenario_tuning.txt
rm -r conf/embeddings_validation.work/m_lgbm__f_mles_emb_tuning_${SC_SUFFIX}_*
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 18 \
    --conf_extra 'report_file: "../results/scenario_tuning.txt", auto_features: ["../data/mles_emb_tuning_*.pickle"]'
less -S results/scenario_tuning.txt


export SC_SUFFIX="optim_sgd_0.001"
rm -r models/mles_ckp_${SC_SUFFIX}
python ../../metric_learning.py \
  params.device="$SC_DEVICE" \
  params.train.optim_type=sgd params.train.lr=0.001 \
  params.train.checkpoints.save_interval=10 \
  params.train.checkpoints.n_saved=1000 \
  params.train.checkpoints.dirname="models/mles_ckp_tuning_${SC_SUFFIX}/" \
  params.train.checkpoints.filename_prefix="mles" \
  params.train.checkpoints.create_dir=true \
  --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050
do
    python ../../ml_inference_lazy.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/mles_ckp_tuning_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_emb_tuning_${SC_SUFFIX}_${SC_EPOCH}" \
        --conf conf/dataset.hocon conf/mles_params.json
done
#rm results/scenario_tuning.txt
#LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
#    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 18 \
#    --conf_extra 'report_file: "../results/scenario_tuning.txt", auto_features: ["../data/mles_emb_tuning_*.pickle"]'
#less -S results/scenario_tuning.txt

export SC_SUFFIX="optim_sgd_0.002"
rm -r models/mles_ckp_${SC_SUFFIX}
python ../../metric_learning.py \
  params.device="$SC_DEVICE" \
  params.train.optim_type=sgd params.train.lr=0.002 \
  params.train.checkpoints.save_interval=10 \
  params.train.checkpoints.n_saved=1000 \
  params.train.checkpoints.dirname="models/mles_ckp_tuning_${SC_SUFFIX}/" \
  params.train.checkpoints.filename_prefix="mles" \
  params.train.checkpoints.create_dir=true \
  --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050
do
    python ../../ml_inference_lazy.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/mles_ckp_tuning_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_emb_tuning_${SC_SUFFIX}_${SC_EPOCH}" \
        --conf conf/dataset.hocon conf/mles_params.json
done
#rm results/scenario_tuning.txt
#LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
#    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 18 \
#    --conf_extra 'report_file: "../results/scenario_tuning.txt", auto_features: ["../data/mles_emb_tuning_*.pickle"]'
#less -S results/scenario_tuning.txt

export SC_SUFFIX="optim_sgd_0.004"
rm -r models/mles_ckp_${SC_SUFFIX}
python ../../metric_learning.py \
  params.device="$SC_DEVICE" \
  params.train.optim_type=sgd params.train.lr=0.004 \
  params.train.checkpoints.save_interval=10 \
  params.train.checkpoints.n_saved=1000 \
  params.train.checkpoints.dirname="models/mles_ckp_tuning_${SC_SUFFIX}/" \
  params.train.checkpoints.filename_prefix="mles" \
  params.train.checkpoints.create_dir=true \
  --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050
do
    python ../../ml_inference_lazy.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/mles_ckp_tuning_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_emb_tuning_${SC_SUFFIX}_${SC_EPOCH}" \
        --conf conf/dataset.hocon conf/mles_params.json
done
#rm results/scenario_tuning.txt
#LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
#    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 18 \
#    --conf_extra 'report_file: "../results/scenario_tuning.txt", auto_features: ["../data/mles_emb_tuning_*.pickle"]'
#less -S results/scenario_tuning.txt

export SC_SUFFIX="optim_sgd_0.008"
rm -r models/mles_ckp_${SC_SUFFIX}
python ../../metric_learning.py \
  params.device="$SC_DEVICE" \
  params.train.optim_type=sgd params.train.lr=0.008 \
  params.train.checkpoints.save_interval=10 \
  params.train.checkpoints.n_saved=1000 \
  params.train.checkpoints.dirname="models/mles_ckp_tuning_${SC_SUFFIX}/" \
  params.train.checkpoints.filename_prefix="mles" \
  params.train.checkpoints.create_dir=true \
  --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050
do
    python ../../ml_inference_lazy.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/mles_ckp_tuning_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_emb_tuning_${SC_SUFFIX}_${SC_EPOCH}" \
        --conf conf/dataset.hocon conf/mles_params.json
done
rm results/scenario_tuning.txt
LUIGI_CONFIG_PATH=conf/luigi.cfg python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 18 \
    --conf_extra 'report_file: "../results/scenario_tuning.txt", auto_features: ["../data/mles_emb_tuning_*.pickle"]'
less -S results/scenario_tuning.txt

