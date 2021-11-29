"""
export LUIGI_CONFIG_PATH=/mnt/data/kireev/pycharm_deploy/embeddings_validation/luigi.cfg

# debug run
# use `--local-schedule` for debug purpose
cd /mnt/data/kireev/pycharm_deploy/embeddings_validation/
rm -r test_conf/train-test.work/; rm test_conf/train-test.txt
PYTHONPATH='.' luigi \
    --workers 5 \
    --module embeddings_validation ReportCollect \
    --conf test_conf/train-test.hocon --total-cpu-count 18
less test_conf/train-test.txt


# production run
cd /mnt/data/kireev/pycharm_deploy/embeddings_validation/
rm -r test_conf/train-test.work/; rm test_conf/train-test.txt
PYTHONPATH="/mnt2/kireev/pycharm-deploy/vector_test" \
    python -m embeddings_validation --workers 1 --conf test_conf/train-test.hocon --total_cpu_count 6 --local_scheduler
less test_conf/train-test.txt

cd /mnt/data/kireev/pycharm_deploy/embeddings_validation/
rm -r test_conf/crossval.work/; rm test_conf/crossval.txt
PYTHONPATH="/mnt/data/kireev/pycharm_1_vec_test" \
    python -m embeddings_validation --workers 1 --conf test_conf/crossval.hocon --total_cpu_count 18
less test_conf/crossval.txt


cd /mnt/data/kireev/pycharm_deploy/embeddings_validation/
rm -r test_conf/single-file.work/; rm test_conf/single-file.txt
PYTHONPATH="/mnt/data/kireev/pycharm_1_vec_test" \
    python -m embeddings_validation --workers 1 --conf test_conf/single-file.hocon --total_cpu_count 18
less test_conf/single-file.txt


cd /mnt/data/kireev/pycharm_deploy/embeddings_validation/
rm -r test_conf/single-file-short.work/; rm test_conf/single-file-short.txt
PYTHONPATH="/mnt/data/kireev/pycharm_1_vec_test" \
    python -m embeddings_validation --workers 1 --conf test_conf/single-file-short.hocon --total_cpu_count 18
less test_conf/single-file-short.txt



cd /mnt/data/kireev/pycharm_1/dltranz/experiments/scenario_gender/
rm -r conf/embeddings_validation.work/; rm results/embeddings_validation.txt
PYTHONPATH="/mnt/data/kireev/pycharm_1_vec_test"   \
    python -m embeddings_validation --workers 10 --conf conf/embeddings_validation.hocon --total_cpu_count 14
less results/embeddings_validation.txt

"""
