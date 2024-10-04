#! /bin/bash

for fold_i in {0..4}
do {
  python /root/pytorch-lifestream/demo/multicoles_test_tools/train_coles.py $fold_i $@ & pid=$!
  PID_LIST+=" $pid";
} done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";
