echo "==== Create data folder ===="

mkdir data 

echo "==== Download from GDrive ===="

gdown 1S-7LUyA-2GY0VHw4wDJJcMCNeOARLcOb

echo "==== Unpacking ===="
unzip alfabattle2b-boosters.pro.zip -d data/

echo "==== Renaming files ===="
mv data/alpha_sample.csv data/sample.csv
mv data/train_transactions_contest data/train_transactions_contest.parquet
mv data/test_transactions_contest data/test_transactions_contest.parquet

echo "==== Removing downloaded archive ===="
rm -r alfabattle2b-boosters.pro.zip 

echo "==== Finish ===="
