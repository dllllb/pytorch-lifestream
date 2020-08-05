# # # # # # # # # # # # # # # #
### MELES sampling N epoch tuning
# # # # # # # # # # # # # # # #


export SC_SUFFIX="sample3_base"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.split_strategy.split_strategy=SampleSlices \
    params.train.split_strategy.cnt_min=25 \
    params.train.split_strategy.cnt_max=75 \
    params.train.n_epoch=200 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150 160 170 180 190 200
do
    python ../../ml_inference.py \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="sample3_mles"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.split_strategy.split_strategy=SampleSlicesMLES \
    params.train.split_strategy.cnt_min=25 \
    params.train.split_strategy.cnt_max=75 \
    params.train.n_epoch=200 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150 160 170 180 190 200
do
    python ../../ml_inference.py \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done


python -m scenario_age_pred compare_approaches --n_workers 5 --models lgb \
    --embedding_file_names "mles_sample3_*.pickle"
                                         oof_accuracy                                                         test_accuracy
                                                 mean t_int_l t_int_h    std                           values          mean t_int_l t_int_h    std                           values
name
lgb_embeds: mles_sample3_base_010.pickle       0.6312  0.6273  0.6351 0.0031  [0.627 0.630 0.631 0.634 0.635]        0.6231  0.6183  0.6279 0.0039  [0.617 0.622 0.624 0.626 0.627]
lgb_embeds: mles_sample3_base_020.pickle       0.6332  0.6292  0.6371 0.0032  [0.628 0.634 0.634 0.635 0.635]        0.6293  0.6267  0.6320 0.0021  [0.627 0.628 0.629 0.630 0.633]
lgb_embeds: mles_sample3_base_030.pickle       0.6348  0.6303  0.6394 0.0037  [0.629 0.635 0.636 0.637 0.638]        0.6314  0.6285  0.6343 0.0023  [0.629 0.630 0.631 0.632 0.635]
lgb_embeds: mles_sample3_base_040.pickle       0.6396  0.6324  0.6468 0.0058  [0.636 0.636 0.637 0.640 0.650]        0.6330  0.6284  0.6376 0.0037  [0.628 0.631 0.634 0.636 0.636]
lgb_embeds: mles_sample3_base_050.pickle       0.6352  0.6318  0.6385 0.0027  [0.630 0.636 0.636 0.636 0.637]        0.6305  0.6273  0.6336 0.0025  [0.627 0.630 0.630 0.631 0.634]
lgb_embeds: mles_sample3_base_060.pickle       0.6369  0.6335  0.6404 0.0028  [0.634 0.635 0.636 0.639 0.640]        0.6333  0.6294  0.6373 0.0032  [0.630 0.630 0.634 0.635 0.638]
lgb_embeds: mles_sample3_base_070.pickle       0.6346  0.6328  0.6364 0.0014  [0.632 0.634 0.635 0.636 0.636]        0.6317  0.6289  0.6344 0.0022  [0.628 0.631 0.632 0.633 0.634]
lgb_embeds: mles_sample3_base_080.pickle       0.6356  0.6315  0.6398 0.0033  [0.633 0.634 0.634 0.638 0.641]        0.6263  0.6212  0.6315 0.0041  [0.621 0.624 0.626 0.628 0.632]
lgb_embeds: mles_sample3_base_090.pickle       0.6363  0.6298  0.6428 0.0052  [0.631 0.632 0.636 0.638 0.644]        0.6277  0.6245  0.6308 0.0025  [0.625 0.626 0.627 0.628 0.632]
lgb_embeds: mles_sample3_base_100.pickle       0.6345  0.6299  0.6390 0.0037  [0.631 0.632 0.634 0.635 0.641]        0.6301  0.6260  0.6343 0.0034  [0.627 0.627 0.630 0.631 0.635]
lgb_embeds: mles_sample3_base_110.pickle       0.6369  0.6348  0.6391 0.0017  [0.635 0.635 0.638 0.638 0.639]        0.6281  0.6253  0.6309 0.0023  [0.625 0.627 0.629 0.629 0.631]
lgb_embeds: mles_sample3_base_120.pickle       0.6344  0.6318  0.6371 0.0021  [0.632 0.634 0.634 0.635 0.638]        0.6275  0.6259  0.6290 0.0013  [0.625 0.627 0.628 0.628 0.628]
lgb_embeds: mles_sample3_base_130.pickle       0.6364  0.6305  0.6424 0.0048  [0.630 0.636 0.636 0.636 0.644]        0.6278  0.6259  0.6297 0.0016  [0.626 0.627 0.628 0.628 0.630]
lgb_embeds: mles_sample3_base_140.pickle       0.6333  0.6287  0.6379 0.0037  [0.629 0.631 0.634 0.636 0.637]        0.6199  0.6155  0.6242 0.0035  [0.614 0.619 0.622 0.622 0.622]
lgb_embeds: mles_sample3_base_150.pickle       0.6327  0.6292  0.6362 0.0029  [0.630 0.631 0.633 0.633 0.637]        0.6291  0.6216  0.6367 0.0061  [0.623 0.626 0.627 0.629 0.639]
lgb_embeds: mles_sample3_base_160.pickle       0.6252  0.6220  0.6284 0.0026  [0.622 0.623 0.625 0.627 0.628]        0.6176  0.6127  0.6225 0.0040  [0.612 0.615 0.617 0.621 0.622]
lgb_embeds: mles_sample3_base_170.pickle       0.6160  0.6119  0.6201 0.0033  [0.613 0.614 0.615 0.618 0.621]        0.6066  0.6039  0.6093 0.0022  [0.604 0.605 0.607 0.608 0.609]
lgb_embeds: mles_sample3_base_180.pickle       0.6066  0.5985  0.6146 0.0065  [0.598 0.603 0.609 0.610 0.614]        0.6044  0.6004  0.6084 0.0033  [0.601 0.603 0.604 0.604 0.610]
lgb_embeds: mles_sample3_base_190.pickle       0.5892  0.5868  0.5917 0.0020  [0.586 0.589 0.590 0.590 0.591]        0.5875  0.5827  0.5923 0.0039  [0.582 0.586 0.587 0.590 0.593]
lgb_embeds: mles_sample3_base_200.pickle       0.6027  0.5969  0.6085 0.0047  [0.595 0.603 0.604 0.604 0.607]        0.6038  0.5963  0.6113 0.0061  [0.594 0.602 0.605 0.609 0.609]

lgb_embeds: mles_sample3_mles_010.pickle       0.6324  0.6244  0.6405 0.0065  [0.628 0.629 0.630 0.631 0.644]        0.6254  0.6216  0.6292 0.0030  [0.621 0.624 0.627 0.627 0.628]
lgb_embeds: mles_sample3_mles_020.pickle       0.6338  0.6284  0.6391 0.0043  [0.627 0.633 0.634 0.637 0.638]        0.6319  0.6295  0.6342 0.0019  [0.629 0.632 0.632 0.633 0.634]
lgb_embeds: mles_sample3_mles_030.pickle       0.6354  0.6307  0.6401 0.0038  [0.633 0.633 0.634 0.637 0.641]        0.6336  0.6303  0.6369 0.0026  [0.630 0.632 0.634 0.636 0.636]
lgb_embeds: mles_sample3_mles_040.pickle       0.6389  0.6339  0.6440 0.0041  [0.636 0.637 0.637 0.639 0.646]        0.6369  0.6302  0.6437 0.0054  [0.629 0.635 0.638 0.639 0.644]
lgb_embeds: mles_sample3_mles_050.pickle       0.6353  0.6330  0.6377 0.0019  [0.634 0.634 0.635 0.636 0.638]        0.6368  0.6312  0.6424 0.0045  [0.631 0.635 0.636 0.638 0.644]
lgb_embeds: mles_sample3_mles_060.pickle       0.6373  0.6323  0.6424 0.0041  [0.633 0.635 0.636 0.640 0.643]        0.6319  0.6269  0.6370 0.0041  [0.628 0.628 0.631 0.635 0.637]
lgb_embeds: mles_sample3_mles_070.pickle       0.6343  0.6293  0.6394 0.0041  [0.629 0.632 0.634 0.637 0.639]        0.6289  0.6230  0.6349 0.0048  [0.623 0.627 0.627 0.632 0.635]
lgb_embeds: mles_sample3_mles_080.pickle       0.6364  0.6304  0.6425 0.0049  [0.632 0.634 0.634 0.639 0.644]        0.6355  0.6313  0.6397 0.0034  [0.630 0.636 0.637 0.637 0.638]
lgb_embeds: mles_sample3_mles_090.pickle       0.6364  0.6317  0.6411 0.0038  [0.631 0.635 0.638 0.638 0.641]        0.6294  0.6262  0.6326 0.0026  [0.626 0.628 0.630 0.631 0.633]
lgb_embeds: mles_sample3_mles_100.pickle       0.6340  0.6282  0.6399 0.0047  [0.629 0.629 0.634 0.638 0.640]        0.6349  0.6295  0.6402 0.0043  [0.631 0.633 0.634 0.635 0.642]
lgb_embeds: mles_sample3_mles_110.pickle       0.6351  0.6319  0.6382 0.0026  [0.633 0.633 0.635 0.636 0.639]        0.6382  0.6344  0.6420 0.0031  [0.634 0.636 0.638 0.641 0.642]
lgb_embeds: mles_sample3_mles_120.pickle       0.6354  0.6321  0.6387 0.0027  [0.631 0.635 0.636 0.637 0.638]        0.6303  0.6274  0.6333 0.0024  [0.628 0.629 0.630 0.630 0.634]
lgb_embeds: mles_sample3_mles_130.pickle       0.6359  0.6309  0.6408 0.0040  [0.632 0.633 0.636 0.636 0.642]        0.6323  0.6290  0.6356 0.0026  [0.630 0.630 0.631 0.635 0.635]
lgb_embeds: mles_sample3_mles_140.pickle       0.6335  0.6302  0.6368 0.0026  [0.630 0.633 0.634 0.634 0.637]        0.6270  0.6211  0.6329 0.0048  [0.622 0.623 0.626 0.629 0.634]
lgb_embeds: mles_sample3_mles_150.pickle       0.6317  0.6280  0.6353 0.0029  [0.626 0.632 0.633 0.633 0.634]        0.6307  0.6248  0.6365 0.0047  [0.625 0.626 0.632 0.635 0.635]
lgb_embeds: mles_sample3_mles_160.pickle       0.6307  0.6288  0.6327 0.0016  [0.628 0.630 0.631 0.632 0.632]        0.6320  0.6271  0.6369 0.0039  [0.629 0.629 0.631 0.634 0.638]
lgb_embeds: mles_sample3_mles_170.pickle       0.6287  0.6217  0.6357 0.0056  [0.622 0.625 0.628 0.633 0.635]        0.6304  0.6246  0.6362 0.0047  [0.625 0.628 0.631 0.631 0.638]
lgb_embeds: mles_sample3_mles_180.pickle       0.6192  0.6154  0.6231 0.0031  [0.616 0.618 0.618 0.619 0.624]        0.6198  0.6174  0.6222 0.0020  [0.618 0.618 0.620 0.621 0.623]
lgb_embeds: mles_sample3_mles_190.pickle       0.6238  0.6222  0.6255 0.0013  [0.622 0.623 0.623 0.625 0.625]        0.6253  0.6195  0.6311 0.0047  [0.622 0.623 0.624 0.625 0.633]
lgb_embeds: mles_sample3_mles_200.pickle       0.6183  0.6122  0.6244 0.0049  [0.612 0.616 0.618 0.620 0.625]        0.6087  0.6033  0.6141 0.0044  [0.603 0.606 0.608 0.612 0.614]
