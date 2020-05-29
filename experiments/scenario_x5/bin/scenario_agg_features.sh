#
# Check features:
# Cat: 'store_id', 'product_id', 'level_1', 'level_2', 'level_3', 'level_4', 'segment_id', 'brand_id', 'vendor_id', 'is_own_trademark', 'is_alcohol'
# Num: 'trn_sum_from_iss', 'product_quantity', 'netto', 'trn_sum_from_red' 'regular_points_received', 'express_points_received', 'regular_points_spent', 'express_points_spent', 'purchase_sum'
#
#
export COL_NUM='"trn_sum_from_iss": "identity"'
for COL_CAT in \
  '"level_1": {"in": 5}' \
  '"level_2": {"in": 45}' \
  '"level_3": {"in": 200}' \
  '"level_4": {"in": 800}' \
  '"segment_id": {"in": 120}' \
  '"brand_id": {"in": 2000}' \
  '"vendor_id": {"in": 1500}' \
  '"is_own_trademark": {"in": 5}' \
  '"is_alcohol": {"in": 5}'
do
  export NAME_CAT=$(echo ${COL_CAT} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')
  export NAME_NUM=$(echo ${COL_NUM} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')

  python ../../metric_learning.py params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    params.train.n_epoch=1 \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    --conf conf/dataset.hocon conf/agg_features_params.json
  python ../../ml_inference.py    params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    output.path="data/scenario__agg_features__${NAME_CAT}_${NAME_NUM}" \
    --conf conf/dataset.hocon conf/agg_features_params.json

done
python -m scenario_x5 compare_approaches --n_workers 1 --models lgb \
  --output_file "results/scenario__agg_features.csv" \
  --embedding_file_names \
  "scenario__agg_features__*.pickle"
#
#
#                                                                                                      oof_rocauc_score                                                         test_rocauc_score
#                                                                                                                  mean t_int_l t_int_h    std                           values              mean t_int_l t_int_h    std                           values
# name
# lgb_embeds: scenario__agg_features__brand_id_in_2000_trn_sum_from_iss_identity.pickle                          0.7542  0.7507  0.7577 0.0025  [0.751 0.752 0.754 0.756 0.757]            0.7550  0.7544  0.7556 0.0004  [0.754 0.755 0.755 0.755 0.756]
# lgb_embeds: scenario__agg_features__is_alcohol_in_5_trn_sum_from_iss_identity.pickle                           0.6049  0.5997  0.6100 0.0037  [0.600 0.602 0.605 0.608 0.609]            0.6032  0.6025  0.6038 0.0005  [0.603 0.603 0.603 0.603 0.604]
# lgb_embeds: scenario__agg_features__is_own_trademark_in_5_trn_sum_from_iss_identity.pickle                     0.6362  0.6299  0.6424 0.0045  [0.632 0.634 0.635 0.636 0.644]            0.6335  0.6327  0.6343 0.0005  [0.633 0.633 0.633 0.634 0.634]
# lgb_embeds: scenario__agg_features__level_1_in_5_trn_sum_from_iss_identity.pickle                              0.6492  0.6402  0.6582 0.0065  [0.639 0.649 0.651 0.651 0.657]            0.6483  0.6476  0.6490 0.0005  [0.648 0.648 0.648 0.649 0.649]
# lgb_embeds: scenario__agg_features__level_2_in_45_trn_sum_from_iss_identity.pickle                             0.7498  0.7436  0.7560 0.0045  [0.744 0.748 0.750 0.751 0.756]            0.7521  0.7516  0.7525 0.0003  [0.752 0.752 0.752 0.752 0.752]
# lgb_embeds: scenario__agg_features__level_3_in_200_trn_sum_from_iss_identity.pickle                            0.7776  0.7730  0.7822 0.0033  [0.774 0.775 0.778 0.779 0.782]            0.7792  0.7787  0.7797 0.0003  [0.779 0.779 0.779 0.779 0.780]
# lgb_embeds: scenario__agg_features__level_4_in_800_trn_sum_from_iss_identity.pickle                            0.7813  0.7766  0.7859 0.0033  [0.777 0.779 0.782 0.783 0.786]            0.7849  0.7840  0.7858 0.0006  [0.784 0.785 0.785 0.785 0.786]
# lgb_embeds: scenario__agg_features__segment_id_in_120_trn_sum_from_iss_identity.pickle                         0.7728  0.7676  0.7780 0.0037  [0.769 0.769 0.774 0.774 0.778]            0.7748  0.7744  0.7751 0.0003  [0.775 0.775 0.775 0.775 0.775]
# lgb_embeds: scenario__agg_features__vendor_id_in_1500_trn_sum_from_iss_identity.pickle                         0.7525  0.7476  0.7573 0.0035  [0.749 0.751 0.751 0.753 0.758]            0.7527  0.7525  0.7530 0.0001  [0.753 0.753 0.753 0.753 0.753]

export COL_NUM='"trn_sum_from_iss": "identity"'
for COL_CAT in \
  '"level_4": {"in": 800}, "level_1": {"in": 5}' \
  '"level_4": {"in": 800}, "level_2": {"in": 45}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}' \
  '"level_4": {"in": 800}, "segment_id": {"in": 120}' \
  '"level_4": {"in": 800}, "brand_id": {"in": 2000}' \
  '"level_4": {"in": 800}, "vendor_id": {"in": 1500}' \
  '"level_4": {"in": 800}, "is_own_trademark": {"in": 5}' \
  '"level_4": {"in": 800}, "is_alcohol": {"in": 5}'
do
  export NAME_CAT=$(echo ${COL_CAT} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')
  export NAME_NUM=$(echo ${COL_NUM} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')

  python ../../metric_learning.py params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    params.train.n_epoch=1 \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    --conf conf/dataset.hocon conf/agg_features_params.json
  python ../../ml_inference.py    params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    output.path="data/scenario__agg_features__${NAME_CAT}_${NAME_NUM}" \
    --conf conf/dataset.hocon conf/agg_features_params.json

done
python -m scenario_x5 compare_approaches --n_workers 1 --models lgb \
  --output_file "results/scenario__agg_features.csv" \
  --embedding_file_names \
  "scenario__agg_features__level_4_in_800*.pickle"
#
#
#                                                                                                      oof_rocauc_score                                                         test_rocauc_score
#                                                                                                                  mean t_int_l t_int_h    std                           values              mean t_int_l t_int_h    std                           values
# name
# lgb_embeds: scenario__agg_features__level_4_in_800_brand_id_in_2000_trn_sum_from_iss_identity.pickle           0.7818  0.7774  0.7862 0.0032  [0.778 0.780 0.782 0.784 0.786]            0.7853  0.7850  0.7856 0.0002  [0.785 0.785 0.785 0.785 0.786]
# lgb_embeds: scenario__agg_features__level_4_in_800_is_alcohol_in_5_trn_sum_from_iss_identity.pickle            0.7814  0.7767  0.7860 0.0034  [0.777 0.779 0.782 0.783 0.786]            0.7851  0.7842  0.7861 0.0007  [0.784 0.785 0.785 0.785 0.786]
# lgb_embeds: scenario__agg_features__level_4_in_800_is_own_trademark_in_5_trn_sum_from_iss_identit...           0.7811  0.7763  0.7859 0.0034  [0.777 0.779 0.782 0.783 0.785]            0.7849  0.7842  0.7856 0.0005  [0.784 0.785 0.785 0.785 0.785]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_1_in_5_trn_sum_from_iss_identity.pickle               0.7821  0.7769  0.7873 0.0037  [0.778 0.779 0.783 0.784 0.787]            0.7845  0.7838  0.7852 0.0005  [0.784 0.784 0.784 0.784 0.785]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_2_in_45_trn_sum_from_iss_identity.pickle              0.7834  0.7781  0.7887 0.0038  [0.780 0.780 0.783 0.785 0.789]            0.7873  0.7867  0.7880 0.0005  [0.787 0.787 0.787 0.788 0.788]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity.pickle             0.7848  0.7805  0.7891 0.0031  [0.781 0.783 0.785 0.787 0.789]            0.7879  0.7873  0.7885 0.0004  [0.788 0.788 0.788 0.788 0.789]
# lgb_embeds: scenario__agg_features__level_4_in_800_segment_id_in_120_trn_sum_from_iss_identity.pi...           0.7847  0.7801  0.7893 0.0033  [0.781 0.783 0.785 0.787 0.789]            0.7874  0.7869  0.7880 0.0004  [0.787 0.787 0.787 0.788 0.788]
# lgb_embeds: scenario__agg_features__level_4_in_800_trn_sum_from_iss_identity.pickle                            0.7813  0.7766  0.7859 0.0033  [0.777 0.779 0.782 0.783 0.786]            0.7849  0.7840  0.7858 0.0006  [0.784 0.785 0.785 0.785 0.786]
# lgb_embeds: scenario__agg_features__level_4_in_800_vendor_id_in_1500_trn_sum_from_iss_identity.pi...           0.7831  0.7788  0.7874 0.0031  [0.779 0.781 0.783 0.785 0.787]            0.7865  0.7858  0.7871 0.0005  [0.786 0.786 0.786 0.787 0.787]

export COL_CAT='"level_4": {"in": 800}'
for COL_NUM in \
  '"trn_sum_from_iss": "identity", "product_quantity": "identity"' \
  '"trn_sum_from_iss": "identity", "netto": "identity"' \
  '"trn_sum_from_iss": "identity", "regular_points_received": "identity"' \
  '"trn_sum_from_iss": "identity", "express_points_received": "identity"' \
  '"trn_sum_from_iss": "identity", "regular_points_spent": "identity"' \
  '"trn_sum_from_iss": "identity", "express_points_spent": "identity"' \
  '"trn_sum_from_iss": "identity", "purchase_sum": "identity"'
do
  export NAME_CAT=$(echo ${COL_CAT} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')
  export NAME_NUM=$(echo ${COL_NUM} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')

  python ../../metric_learning.py params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    params.train.n_epoch=1 \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    --conf conf/dataset.hocon conf/agg_features_params.json
  python ../../ml_inference.py    params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    output.path="data/scenario__agg_features__${NAME_CAT}_${NAME_NUM}" \
    --conf conf/dataset.hocon conf/agg_features_params.json

done
python -m scenario_x5 compare_approaches --n_workers 1 --models lgb \
  --output_file "results/scenario__agg_features.csv" \
  --embedding_file_names \
  "scenario__agg_features__level_4_in_800_trn_sum_from_iss_identity*.pickle"
#
#
#                                                                                                      oof_rocauc_score                                                         test_rocauc_score
#                                                                                                                  mean t_int_l t_int_h    std                           values              mean t_int_l t_int_h    std                           values
# name
# lgb_embeds: scenario__agg_features__level_4_in_800_trn_sum_from_iss_identity_express_points_recei...           0.7812  0.7767  0.7857 0.0033  [0.777 0.779 0.782 0.783 0.785]            0.7849  0.7839  0.7860 0.0007  [0.784 0.785 0.785 0.786 0.786]
# lgb_embeds: scenario__agg_features__level_4_in_800_trn_sum_from_iss_identity_express_points_spent...           0.7815  0.7767  0.7863 0.0035  [0.777 0.779 0.782 0.783 0.786]            0.7852  0.7843  0.7861 0.0006  [0.784 0.785 0.785 0.786 0.786]
# lgb_embeds: scenario__agg_features__level_4_in_800_trn_sum_from_iss_identity_netto_identity.pickle             0.7831  0.7792  0.7871 0.0028  [0.779 0.781 0.784 0.784 0.786]            0.7858  0.7853  0.7863 0.0004  [0.785 0.786 0.786 0.786 0.786]
# lgb_embeds: scenario__agg_features__level_4_in_800_trn_sum_from_iss_identity_product_quantity_ide...           0.7816  0.7767  0.7865 0.0035  [0.777 0.779 0.783 0.783 0.786]            0.7852  0.7844  0.7859 0.0005  [0.785 0.785 0.785 0.785 0.786]
# lgb_embeds: scenario__agg_features__level_4_in_800_trn_sum_from_iss_identity_purchase_sum_identit...           0.7820  0.7772  0.7867 0.0034  [0.778 0.780 0.782 0.784 0.786]            0.7858  0.7847  0.7870 0.0008  [0.785 0.786 0.786 0.786 0.787]
# lgb_embeds: scenario__agg_features__level_4_in_800_trn_sum_from_iss_identity_regular_points_recei...           0.7819  0.7771  0.7867 0.0035  [0.778 0.780 0.782 0.784 0.786]            0.7858  0.7848  0.7867 0.0007  [0.785 0.785 0.786 0.786 0.787]
# lgb_embeds: scenario__agg_features__level_4_in_800_trn_sum_from_iss_identity_regular_points_spent...           0.7818  0.7770  0.7866 0.0035  [0.777 0.780 0.782 0.783 0.786]            0.7851  0.7841  0.7861 0.0007  [0.784 0.785 0.785 0.786 0.786]


export COL_NUM='"trn_sum_from_iss": "identity"'
for COL_CAT in \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "level_1": {"in": 5}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "level_2": {"in": 45}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "segment_id": {"in": 120}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "brand_id": {"in": 2000}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "vendor_id": {"in": 1500}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "is_own_trademark": {"in": 5}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "is_alcohol": {"in": 5}'
do
  export NAME_CAT=$(echo ${COL_CAT} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')
  export NAME_NUM=$(echo ${COL_NUM} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')

  python ../../metric_learning.py params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    params.train.n_epoch=1 \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    --conf conf/dataset.hocon conf/agg_features_params.json
  python ../../ml_inference.py    params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    output.path="data/scenario__agg_features__${NAME_CAT}_${NAME_NUM}" \
    --conf conf/dataset.hocon conf/agg_features_params.json

done
python -m scenario_x5 compare_approaches --n_workers 1 --models lgb \
  --output_file "results/scenario__agg_features.csv" \
  --embedding_file_names \
  "scenario__agg_features__level_4_in_800_level_3_in_200*.pickle"
#                                                                                                                                     oof_rocauc_score                                                         test_rocauc_score
#                                                                                                                                                 mean t_int_l t_int_h    std                           values              mean t_int_l t_int_h    std                           values
# name
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity.pickle                                            0.7848  0.7805  0.7891 0.0031  [0.781 0.783 0.785 0.787 0.789]            0.7879  0.7873  0.7885 0.0004  [0.788 0.788 0.788 0.788 0.789]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_brand_id_in_2000_trn_sum_from_iss_identity.pickle                           0.7857  0.7815  0.7899 0.0030  [0.782 0.784 0.786 0.788 0.790]            0.7884  0.7881  0.7887 0.0002  [0.788 0.788 0.788 0.789 0.789]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_is_alcohol_in_5_trn_sum_from_iss_identity.pickle                            0.7851  0.7810  0.7892 0.0029  [0.781 0.783 0.785 0.787 0.789]            0.7881  0.7876  0.7886 0.0003  [0.788 0.788 0.788 0.788 0.789]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_is_own_trademark_in_5_trn_sum_from_iss_identity.pickle                      0.7849  0.7809  0.7890 0.0029  [0.781 0.783 0.785 0.786 0.789]            0.7879  0.7874  0.7883 0.0003  [0.787 0.788 0.788 0.788 0.788]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_level_1_in_5_trn_sum_from_iss_identity.pickle                               0.7850  0.7806  0.7894 0.0032  [0.781 0.783 0.785 0.786 0.789]            0.7875  0.7866  0.7883 0.0006  [0.787 0.787 0.787 0.788 0.788]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_level_2_in_45_trn_sum_from_iss_identity.pickle                              0.7855  0.7807  0.7904 0.0035  [0.782 0.782 0.786 0.788 0.790]            0.7891  0.7883  0.7899 0.0006  [0.788 0.789 0.789 0.789 0.790]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_segment_id_in_120_trn_sum_from_iss_identity.pickle                          0.7860  0.7817  0.7903 0.0031  [0.782 0.784 0.786 0.788 0.790]            0.7887  0.7882  0.7892 0.0004  [0.788 0.788 0.789 0.789 0.789]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_vendor_id_in_1500_trn_sum_from_iss_identity.pickle                          0.7861  0.7820  0.7901 0.0029  [0.783 0.784 0.786 0.788 0.790]            0.7888  0.7880  0.7896 0.0006  [0.788 0.788 0.789 0.789 0.790]



export COL_CAT='"level_4": {"in": 800}, "level_3": {"in": 200}'
for COL_NUM in \
  '"trn_sum_from_iss": "identity", "product_quantity": "identity"' \
  '"trn_sum_from_iss": "identity", "netto": "identity"' \
  '"trn_sum_from_iss": "identity", "regular_points_received": "identity"' \
  '"trn_sum_from_iss": "identity", "express_points_received": "identity"' \
  '"trn_sum_from_iss": "identity", "regular_points_spent": "identity"' \
  '"trn_sum_from_iss": "identity", "express_points_spent": "identity"' \
  '"trn_sum_from_iss": "identity", "purchase_sum": "identity"'
do
  export NAME_CAT=$(echo ${COL_CAT} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')
  export NAME_NUM=$(echo ${COL_NUM} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')

  python ../../metric_learning.py params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    params.train.n_epoch=1 \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    --conf conf/dataset.hocon conf/agg_features_params.json
  python ../../ml_inference.py    params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    output.path="data/scenario__agg_features__${NAME_CAT}_${NAME_NUM}" \
    --conf conf/dataset.hocon conf/agg_features_params.json

done
python -m scenario_x5 compare_approaches --n_workers 1 --models lgb \
  --output_file "results/scenario__agg_features.csv" \
  --embedding_file_names \
  "scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity*.pickle"
#                                                                                                                                     oof_rocauc_score                                                         test_rocauc_score
#                                                                                                                                                 mean t_int_l t_int_h    std                           values              mean t_int_l t_int_h    std                           values
# name
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity.pickle                                            0.7848  0.7805  0.7891 0.0031  [0.781 0.783 0.785 0.787 0.789]            0.7879  0.7873  0.7885 0.0004  [0.788 0.788 0.788 0.788 0.789]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity_express_points_received_identity.pickle           0.7851  0.7807  0.7894 0.0031  [0.781 0.783 0.785 0.787 0.789]            0.7883  0.7876  0.7890 0.0005  [0.788 0.788 0.788 0.788 0.789]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity_express_points_spent_identity.pickle              0.7852  0.7808  0.7896 0.0032  [0.781 0.783 0.785 0.787 0.789]            0.7884  0.7880  0.7889 0.0003  [0.788 0.788 0.788 0.789 0.789]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity_netto_identity.pickle                             0.7871  0.7837  0.7905 0.0024  [0.784 0.786 0.787 0.788 0.790]            0.7891  0.7883  0.7898 0.0005  [0.788 0.789 0.789 0.789 0.790]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity_product_quantity_identity.pickle                  0.7855  0.7816  0.7894 0.0028  [0.782 0.784 0.786 0.787 0.789]            0.7885  0.7877  0.7892 0.0005  [0.788 0.788 0.788 0.789 0.789]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity_purchase_sum_identity.pickle                      0.7858  0.7816  0.7901 0.0031  [0.782 0.784 0.786 0.788 0.789]            0.7889  0.7879  0.7898 0.0007  [0.788 0.788 0.789 0.789 0.790]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity_regular_points_received_identity.pickle           0.7859  0.7815  0.7903 0.0031  [0.782 0.784 0.786 0.788 0.790]            0.7889  0.7883  0.7895 0.0004  [0.788 0.788 0.789 0.789 0.789]
# lgb_embeds: scenario__agg_features__level_4_in_800_level_3_in_200_trn_sum_from_iss_identity_regular_points_spent_identity.pickle              0.7860  0.7819  0.7901 0.0029  [0.782 0.784 0.786 0.788 0.790]            0.7886  0.7882  0.7890 0.0003  [0.788 0.788 0.789 0.789 0.789]

export COL_NUM='"trn_sum_from_iss": "identity", "netto": "identity"'
for COL_CAT in \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "level_1": {"in": 5}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "level_2": {"in": 45}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "segment_id": {"in": 120}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "brand_id": {"in": 2000}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "vendor_id": {"in": 1500}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "is_own_trademark": {"in": 5}' \
  '"level_4": {"in": 800}, "level_3": {"in": 200}, "is_alcohol": {"in": 5}'
do
  export NAME_CAT=$(echo ${COL_CAT} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')
  export NAME_NUM=$(echo ${COL_NUM} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')

  python ../../metric_learning.py params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    params.train.n_epoch=1 \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    --conf conf/dataset.hocon conf/agg_features_params.json
  python ../../ml_inference.py    params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    output.path="data/scenario__agg_features__${NAME_CAT}_${NAME_NUM}" \
    --conf conf/dataset.hocon conf/agg_features_params.json

done
export COL_CAT='"level_4": {"in": 800}, "level_3": {"in": 200}'
for COL_NUM in \
  '"trn_sum_from_iss": "identity", "netto": "identity", "product_quantity": "identity"' \
  '"trn_sum_from_iss": "identity", "netto": "identity", "regular_points_received": "identity"' \
  '"trn_sum_from_iss": "identity", "netto": "identity", "express_points_received": "identity"' \
  '"trn_sum_from_iss": "identity", "netto": "identity", "regular_points_spent": "identity"' \
  '"trn_sum_from_iss": "identity", "netto": "identity", "express_points_spent": "identity"' \
  '"trn_sum_from_iss": "identity", "netto": "identity", "purchase_sum": "identity"'
do
  export NAME_CAT=$(echo ${COL_CAT} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')
  export NAME_NUM=$(echo ${COL_NUM} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')

  python ../../metric_learning.py params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    params.train.n_epoch=1 \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    --conf conf/dataset.hocon conf/agg_features_params.json
  python ../../ml_inference.py    params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    output.path="data/scenario__agg_features__${NAME_CAT}_${NAME_NUM}" \
    --conf conf/dataset.hocon conf/agg_features_params.json

done
python -m scenario_x5 compare_approaches --n_workers 2 --models lgb \
  --output_file "results/scenario__agg_features.csv" \
  --embedding_file_names \
  "scenario__agg_features__level_4_in_800_level_3_in_200*.pickle"



export COL_NUM=''
for COL_CAT in \
  '"level_1": {"in": 5}' \
  '"level_2": {"in": 45}' \
  '"brand_id": {"in": 2000}' \
  '"vendor_id": {"in": 1500}' \
  '"is_own_trademark": {"in": 5}' \
  '"is_alcohol": {"in": 5}'
do
  export NAME_CAT=$(echo ${COL_CAT} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')
  export NAME_NUM=$(echo ${COL_NUM} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')

  python ../../metric_learning.py params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    params.train.n_epoch=1 dataset.max_rows=50000 \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    --conf conf/dataset.hocon conf/agg_features_params.json
  python ../../ml_inference.py    params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    output.path="data/scenario__agg_features__${NAME_CAT}_${NAME_NUM}" \
    --conf conf/dataset.hocon conf/agg_features_params.json
done
export COL_CAT=''
for COL_NUM in \
  '"product_quantity": "identity"' \
  '"express_points_received": "identity"' \
  '"regular_points_spent": "identity"' \
  '"express_points_spent": "identity"' \
  '"purchase_sum": "identity"'
do
  export NAME_CAT=$(echo ${COL_CAT} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')
  export NAME_NUM=$(echo ${COL_NUM} | sed -e 's/[\:\,]/\_/g' | sed -e 's/[^A-Za-z0-9._-]//g')

  python ../../metric_learning.py params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    params.train.n_epoch=1 dataset.max_rows=50000 \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    --conf conf/dataset.hocon conf/agg_features_params.json
  python ../../ml_inference.py    params.device="$SC_DEVICE" \
    params.trx_encoder.embeddings="{${COL_CAT}}" \
    params.trx_encoder.numeric_values="{${COL_NUM}}" \
    model_path.model="models/scenario__agg_features__${NAME_CAT}_${NAME_NUM}.p" \
    output.path="data/scenario__agg_features__${NAME_CAT}_${NAME_NUM}" \
    --conf conf/dataset.hocon conf/agg_features_params.json
done
python -m scenario_x5 compare_approaches --n_workers 2 --models lgb \
  --output_file "results/scenario__agg_features.csv" \
  --embedding_file_names \
  "scenario__agg_features*.pickle"
