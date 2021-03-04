# # # # # # # # # # # # # # # #
### MELES sampling LR tuning
# # # # # # # # # # # # # # # #


export SC_SUFFIX="sample5_base"
for SC_LR in "0.0005" "0.0010" "0.0020" "0.0040" "0.0080"
do
    python ../../metric_learning.py \
        params.device="$SC_DEVICE" \
        params.train.split_strategy.split_strategy=SampleSlices \
        params.train.split_strategy.cnt_min=25 \
        params.train.split_strategy.cnt_max=200 \
        params.train.n_epoch=100 \
        params.train.lr=${SC_LR} \
        model_path.model="models/mles_model_${SC_SUFFIX}_${SC_LR}.p" \
        params.train.checkpoints.save_interval=10 \
        params.train.checkpoints.n_saved=1000 \
        params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}_${SC_LR}/" \
        params.train.checkpoints.filename_prefix="mles" \
        params.train.checkpoints.create_dir=true \
        --conf conf/dataset.hocon conf/mles_params.json
    for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100
    do
        python ../../ml_inference.py \
            model_path.model="models/mles_checkpoints_${SC_SUFFIX}_${SC_LR}/mles_model_${SC_EPOCH##+(0)}.pt" \
            output.path="data/mles_${SC_SUFFIX}_${SC_LR}_${SC_EPOCH}" \
            params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
    done
done


export SC_SUFFIX="sample5_mles"
for SC_LR in "0.0005" "0.0010" "0.0020" "0.0040" "0.0080"
do
    python ../../metric_learning.py \
        params.device="$SC_DEVICE" \
        params.train.split_strategy.split_strategy=SampleSlicesMLES \
        params.train.split_strategy.cnt_min=25 \
        params.train.split_strategy.cnt_max=200 \
        params.train.n_epoch=100 \
        params.train.lr=${SC_LR} \
        model_path.model="models/mles_model_${SC_SUFFIX}_${SC_LR}.p" \
        params.train.checkpoints.save_interval=10 \
        params.train.checkpoints.n_saved=1000 \
        params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}_${SC_LR}/" \
        params.train.checkpoints.filename_prefix="mles" \
        params.train.checkpoints.create_dir=true \
        --conf conf/dataset.hocon conf/mles_params.json
    for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100
    do
        python ../../ml_inference.py \
            model_path.model="models/mles_checkpoints_${SC_SUFFIX}_${SC_LR}/mles_model_${SC_EPOCH##+(0)}.pt" \
            output.path="data/mles_${SC_SUFFIX}_${SC_LR}_${SC_EPOCH}" \
            params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
    done
done

python -m scenario_age_pred compare_approaches --n_workers 5 --models lgb \
    --embedding_file_names "mles_sample5_*.pickle"


                                                oof_accuracy                                                         test_accuracy
                                                        mean t_int_l t_int_h    std                           values          mean t_int_l t_int_h    std                           values
name
lgb_embeds: mles_sample5_base_0.0005_040.pickle       0.6281  0.6219  0.6343 0.0050  [0.623 0.625 0.628 0.628 0.636]        0.6292  0.6258  0.6326 0.0028  [0.626 0.627 0.630 0.630 0.633]
lgb_embeds: mles_sample5_base_0.0010_040.pickle       0.6299  0.6265  0.6333 0.0027  [0.627 0.629 0.629 0.630 0.634]        0.6331  0.6292  0.6370 0.0031  [0.630 0.631 0.632 0.634 0.638]
lgb_embeds: mles_sample5_base_0.0020_040.pickle       0.6344  0.6275  0.6414 0.0056  [0.626 0.633 0.635 0.637 0.642]        0.6324  0.6269  0.6379 0.0044  [0.628 0.630 0.631 0.634 0.639]
lgb_embeds: mles_sample5_base_0.0040_040.pickle       0.6217  0.6173  0.6261 0.0036  [0.618 0.620 0.621 0.622 0.627]        0.6264  0.6188  0.6340 0.0061  [0.617 0.625 0.627 0.630 0.633]
lgb_embeds: mles_sample5_base_0.0080_040.pickle       0.5347  0.5280  0.5413 0.0053  [0.529 0.531 0.533 0.539 0.542]        0.5318  0.5272  0.5364 0.0037  [0.526 0.532 0.532 0.533 0.536]

lgb_embeds: mles_sample5_mles_0.0005_040.pickle       0.6257  0.6207  0.6306 0.0040  [0.619 0.625 0.628 0.628 0.629]        0.6279  0.6253  0.6305 0.0021  [0.626 0.626 0.626 0.630 0.630]
lgb_embeds: mles_sample5_mles_0.0010_040.pickle       0.6301  0.6200  0.6402 0.0081  [0.624 0.624 0.627 0.633 0.643]        0.6333  0.6301  0.6365 0.0026  [0.630 0.631 0.634 0.635 0.636]
lgb_embeds: mles_sample5_mles_0.0020_040.pickle       0.6344  0.6287  0.6400 0.0046  [0.629 0.632 0.634 0.635 0.642]        0.6379  0.6342  0.6417 0.0030  [0.634 0.636 0.639 0.640 0.641]
lgb_embeds: mles_sample5_mles_0.0040_040.pickle       0.6209  0.6179  0.6238 0.0024  [0.619 0.619 0.620 0.622 0.624]        0.6321  0.6272  0.6369 0.0039  [0.626 0.631 0.633 0.633 0.637]
lgb_embeds: mles_sample5_mles_0.0080_040.pickle       0.5443  0.5379  0.5508 0.0052  [0.537 0.543 0.543 0.549 0.550]        0.5462  0.5357  0.5567 0.0085  [0.534 0.544 0.544 0.553 0.556]


lgb_embeds: mles_sample5_base_0.0005_080.pickle       0.6332  0.6248  0.6417 0.0068  [0.623 0.631 0.634 0.638 0.640]        0.6322  0.6239  0.6405 0.0067  [0.624 0.627 0.633 0.637 0.640]
lgb_embeds: mles_sample5_base_0.0010_080.pickle       0.6325  0.6284  0.6366 0.0033  [0.627 0.632 0.633 0.634 0.636]        0.6361  0.6331  0.6390 0.0024  [0.633 0.635 0.637 0.637 0.639]
lgb_embeds: mles_sample5_base_0.0020_080.pickle       0.6363  0.6306  0.6419 0.0046  [0.630 0.634 0.636 0.639 0.642]        0.6377  0.6348  0.6407 0.0023  [0.635 0.636 0.639 0.639 0.641]
lgb_embeds: mles_sample5_base_0.0040_080.pickle       0.6186  0.6154  0.6218 0.0026  [0.616 0.617 0.619 0.619 0.623]        0.6271  0.6238  0.6304 0.0026  [0.623 0.627 0.627 0.628 0.631]
lgb_embeds: mles_sample5_base_0.0080_080.pickle       0.5374  0.5316  0.5432 0.0047  [0.532 0.535 0.537 0.540 0.544]        0.5334  0.5322  0.5346 0.0010  [0.533 0.533 0.533 0.533 0.535]

lgb_embeds: mles_sample5_mles_0.0005_080.pickle       0.6297  0.6233  0.6362 0.0052  [0.621 0.629 0.630 0.634 0.634]        0.6322  0.6273  0.6371 0.0040  [0.626 0.631 0.633 0.635 0.636]
lgb_embeds: mles_sample5_mles_0.0010_080.pickle       0.6378  0.6291  0.6465 0.0070  [0.631 0.634 0.637 0.638 0.649]        0.6395  0.6336  0.6453 0.0047  [0.635 0.637 0.638 0.640 0.647]
lgb_embeds: mles_sample5_mles_0.0020_080.pickle       0.6374  0.6319  0.6428 0.0044  [0.633 0.635 0.636 0.639 0.644]        0.6415  0.6371  0.6458 0.0035  [0.639 0.640 0.640 0.641 0.648]
lgb_embeds: mles_sample5_mles_0.0040_080.pickle       0.6195  0.6141  0.6248 0.0043  [0.615 0.617 0.619 0.620 0.626]        0.6342  0.6320  0.6364 0.0017  [0.633 0.633 0.634 0.635 0.637]
lgb_embeds: mles_sample5_mles_0.0080_080.pickle       0.5441  0.5395  0.5488 0.0038  [0.539 0.542 0.546 0.547 0.547]        0.5443  0.5362  0.5524 0.0065  [0.535 0.540 0.546 0.550 0.550]


lgb_embeds: mles_sample5_base_0.0005_100.pickle       0.6326  0.6268  0.6383 0.0046  [0.625 0.632 0.634 0.635 0.636]        0.6328  0.6312  0.6344 0.0013  [0.631 0.632 0.633 0.634 0.634]
lgb_embeds: mles_sample5_base_0.0010_100.pickle       0.6377  0.6349  0.6405 0.0023  [0.636 0.636 0.637 0.639 0.641]        0.6356  0.6294  0.6418 0.0050  [0.628 0.635 0.636 0.636 0.642]
lgb_embeds: mles_sample5_base_0.0020_100.pickle       0.6351  0.6303  0.6399 0.0039  [0.631 0.632 0.635 0.637 0.641]        0.6439  0.6396  0.6482 0.0035  [0.641 0.642 0.643 0.644 0.650]
lgb_embeds: mles_sample5_base_0.0040_100.pickle       0.5778  0.5746  0.5809 0.0025  [0.574 0.577 0.578 0.580 0.581]        0.5821  0.5755  0.5887 0.0053  [0.576 0.577 0.585 0.586 0.587]
lgb_embeds: mles_sample5_base_0.0080_100.pickle       0.5385  0.5286  0.5484 0.0080  [0.529 0.535 0.535 0.546 0.547]        0.5373  0.5331  0.5414 0.0033  [0.534 0.535 0.536 0.541 0.541]

lgb_embeds: mles_sample5_mles_0.0005_100.pickle       0.6319  0.6267  0.6371 0.0042  [0.625 0.632 0.633 0.633 0.637]        0.6348  0.6268  0.6428 0.0065  [0.629 0.630 0.631 0.641 0.643]
lgb_embeds: mles_sample5_mles_0.0010_100.pickle       0.6381  0.6337  0.6425 0.0035  [0.634 0.636 0.637 0.640 0.643]        0.6359  0.6331  0.6386 0.0022  [0.633 0.635 0.636 0.638 0.638]
lgb_embeds: mles_sample5_mles_0.0020_100.pickle       0.6393  0.6331  0.6454 0.0049  [0.633 0.636 0.639 0.642 0.646]        0.6412  0.6375  0.6449 0.0030  [0.637 0.641 0.641 0.642 0.645]
lgb_embeds: mles_sample5_mles_0.0040_100.pickle       0.6207  0.6133  0.6282 0.0060  [0.615 0.615 0.619 0.627 0.627]        0.6312  0.6272  0.6352 0.0032  [0.627 0.630 0.631 0.633 0.636]
lgb_embeds: mles_sample5_mles_0.0080_100.pickle       0.5410  0.5358  0.5461 0.0041  [0.536 0.537 0.542 0.544 0.545]        0.5405  0.5360  0.5450 0.0036  [0.538 0.538 0.539 0.541 0.547]
