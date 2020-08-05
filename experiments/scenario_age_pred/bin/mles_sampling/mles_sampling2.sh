# # # # # # # # # # # # # # # #
### MELES sampling cnt_max tuning
# # # # # # # # # # # # # # # #


export SC_SUFFIX="sample2_base"
for SC_MIN in 0025
do
  for SC_MAX in 0600 0800
  do
    python ../../metric_learning.py \
        params.device="$SC_DEVICE" \
        params.train.split_strategy.cnt_min=${SC_MIN} \
        params.train.split_strategy.cnt_max=${SC_MAX} \
        model_path.model="models/mles_model_${SC_SUFFIX}_${SC_MIN}_${SC_MAX}.p" \
        output.path="data/mles_embeddings_${SC_SUFFIX}_${SC_MIN}_${SC_MAX}" \
        --conf conf/dataset.hocon conf/mles_params.json
    python ../../ml_inference.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/mles_model_${SC_SUFFIX}_${SC_MIN}_${SC_MAX}.p" \
        output.path="data/mles_embeddings_${SC_SUFFIX}_${SC_MIN}_${SC_MAX}" \
        --conf conf/dataset.hocon conf/mles_params.json
  done
done

export SC_SUFFIX="sample2_mles"
for SC_MIN in 0025
do
  for SC_MAX in 0600 0800
  do
    python ../../metric_learning.py \
        params.device="$SC_DEVICE" \
        params.train.split_strategy.split_strategy="SampleSlicesMLES" \
        params.train.split_strategy.cnt_min=${SC_MIN} \
        params.train.split_strategy.cnt_max=${SC_MAX} \
        model_path.model="models/mles_model_${SC_SUFFIX}_${SC_MIN}_${SC_MAX}.p" \
        output.path="data/mles_embeddings_${SC_SUFFIX}_${SC_MIN}_${SC_MAX}" \
        --conf conf/dataset.hocon conf/mles_params.json
    python ../../ml_inference.py \
        params.device="$SC_DEVICE" \
        model_path.model="models/mles_model_${SC_SUFFIX}_${SC_MIN}_${SC_MAX}.p" \
        output.path="data/mles_embeddings_${SC_SUFFIX}_${SC_MIN}_${SC_MAX}" \
        --conf conf/dataset.hocon conf/mles_params.json
  done
done

python -m scenario_age_pred compare_approaches --n_workers 5 --models lgb \
    --embedding_file_names "mles_embeddings_sample2_*.pickle"

lgb_embeds: mles_embeddings_sample2_base_0025_0075.pickle       0.6345  0.6299  0.6390 0.0037  [0.631 0.632 0.634 0.635 0.641]        0.6301  0.6260  0.6343 0.0034  [0.627 0.627 0.630 0.631 0.635]
lgb_embeds: mles_embeddings_sample2_base_0025_0100.pickle       0.6319  0.6269  0.6369 0.0040  [0.627 0.629 0.631 0.634 0.638]        0.6315  0.6263  0.6366 0.0041  [0.627 0.629 0.631 0.632 0.638]
lgb_embeds: mles_embeddings_sample2_base_0025_0125.pickle       0.6368  0.6331  0.6405 0.0030  [0.634 0.635 0.636 0.637 0.642]        0.6255  0.6200  0.6311 0.0045  [0.620 0.623 0.625 0.627 0.632]
lgb_embeds: mles_embeddings_sample2_base_0025_0150.pickle       0.6368  0.6341  0.6396 0.0022  [0.634 0.635 0.638 0.638 0.639]        0.6411  0.6379  0.6443 0.0026  [0.638 0.640 0.640 0.643 0.644]
lgb_embeds: mles_embeddings_sample2_base_0025_0175.pickle       0.6330  0.6275  0.6386 0.0045  [0.627 0.630 0.636 0.636 0.637]        0.6379  0.6332  0.6426 0.0038  [0.632 0.636 0.640 0.641 0.641]
lgb_embeds: mles_embeddings_sample2_base_0025_0200.pickle       0.6351  0.6303  0.6399 0.0039  [0.631 0.632 0.635 0.637 0.641]        0.6439  0.6396  0.6482 0.0035  [0.641 0.642 0.643 0.644 0.650]
lgb_embeds: mles_embeddings_sample2_base_0025_0250.pickle       0.5235  0.5126  0.5343 0.0087  [0.512 0.520 0.524 0.525 0.536]        0.5181  0.5144  0.5217 0.0029  [0.514 0.517 0.518 0.519 0.522]
lgb_embeds: mles_embeddings_sample2_base_0025_0300.pickle       0.5958  0.5899  0.6017 0.0047  [0.592 0.592 0.594 0.599 0.603]        0.5911  0.5872  0.5951 0.0032  [0.587 0.589 0.591 0.594 0.594]
lgb_embeds: mles_embeddings_sample2_base_0025_0350.pickle       0.6282  0.6225  0.6339 0.0046  [0.622 0.626 0.629 0.630 0.634]        0.6334  0.6290  0.6378 0.0036  [0.630 0.632 0.633 0.633 0.639]
lgb_embeds: mles_embeddings_sample2_base_0025_0400.pickle       0.6244  0.6218  0.6270 0.0021  [0.622 0.624 0.624 0.625 0.628]        0.6352  0.6283  0.6421 0.0056  [0.630 0.633 0.633 0.636 0.644]
###
lgb_embeds: mles_embeddings_sample2_base_0025_0800.pickle       0.5659  0.5601  0.5717 0.0047  [0.560 0.562 0.567 0.569 0.571]        0.5656  0.5577  0.5735 0.0063  [0.560 0.562 0.564 0.566 0.576]

lgb_embeds: mles_embeddings_sample2_mles_0025_0075.pickle       0.6340  0.6282  0.6399 0.0047  [0.629 0.629 0.634 0.638 0.640]        0.6349  0.6295  0.6402 0.0043  [0.631 0.633 0.634 0.635 0.642]
lgb_embeds: mles_embeddings_sample2_mles_0025_0100.pickle       0.6329  0.6271  0.6387 0.0047  [0.626 0.631 0.632 0.637 0.638]        0.6292  0.6254  0.6330 0.0030  [0.625 0.627 0.630 0.632 0.632]
lgb_embeds: mles_embeddings_sample2_mles_0025_0125.pickle       0.6380  0.6343  0.6418 0.0030  [0.634 0.636 0.639 0.640 0.641]        0.6361  0.6293  0.6430 0.0055  [0.628 0.635 0.638 0.638 0.643]
lgb_embeds: mles_embeddings_sample2_mles_0025_0150.pickle       0.6361  0.6279  0.6442 0.0065  [0.625 0.637 0.637 0.640 0.641]        0.6394  0.6371  0.6417 0.0018  [0.637 0.638 0.640 0.640 0.642]
lgb_embeds: mles_embeddings_sample2_mles_0025_0175.pickle       0.6365  0.6333  0.6396 0.0025  [0.634 0.635 0.636 0.636 0.641]        0.6356  0.6340  0.6372 0.0013  [0.634 0.635 0.636 0.637 0.637]
lgb_embeds: mles_embeddings_sample2_mles_0025_0200.pickle       0.6393  0.6331  0.6454 0.0049  [0.633 0.636 0.639 0.642 0.646]        0.6412  0.6375  0.6449 0.0030  [0.637 0.641 0.641 0.642 0.645]
lgb_embeds: mles_embeddings_sample2_mles_0025_0250.pickle       0.6318  0.6245  0.6391 0.0059  [0.626 0.627 0.631 0.634 0.641]        0.6303  0.6277  0.6330 0.0021  [0.628 0.628 0.631 0.632 0.633]
lgb_embeds: mles_embeddings_sample2_mles_0025_0300.pickle       0.6354  0.6279  0.6428 0.0060  [0.628 0.632 0.636 0.637 0.644]        0.6360  0.6290  0.6430 0.0056  [0.628 0.634 0.637 0.639 0.642]
lgb_embeds: mles_embeddings_sample2_mles_0025_0350.pickle       0.6269  0.6226  0.6313 0.0035  [0.624 0.624 0.627 0.628 0.632]        0.6296  0.6229  0.6363 0.0054  [0.625 0.626 0.628 0.630 0.639]
lgb_embeds: mles_embeddings_sample2_mles_0025_0400.pickle       0.6361  0.6281  0.6442 0.0065  [0.627 0.634 0.636 0.639 0.644]        0.6409  0.6380  0.6437 0.0023  [0.638 0.640 0.641 0.642 0.644]
lgb_embeds: mles_embeddings_sample2_mles_0025_0600.pickle       0.6311  0.6255  0.6368 0.0046  [0.625 0.627 0.634 0.634 0.636]        0.6413  0.6352  0.6475 0.0049  [0.634 0.640 0.642 0.645 0.646]
###

python -m scenario_age_pred compare_approaches --n_workers 5 --models lgb \
    --embedding_file_names "mles_embeddings_sample2_*_0025_0[6-8]??.pickle"
