import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def transform(x):
    return np.sign(x) * np.log(np.abs(x) + 1)

def transform_inv(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

def create_new_targets_on_gender_train_csv(file_name_in, filename_out, TR_AMOUNTS_COL=1,
                                           TR_TYPES_COL=4, top_THR=5, take_first_fraction=0.5):
    '''
    This function changes target to spending/income distribution and write it to `filename_out`.
    filename_in : parquet file name
    filename_out : `gender_train.csv`-like file name
    TR_AMOUNTS_COL : column index with amounts of value in transactions
    TR_TYPES_COL : column index with types of transactions
    top_THR : number of the most valuable transactions chosen to create classes for target,
              total number of classes equal to `top_THR` + 1 (the last class for others)
    take_first_fraction: control the fraction of transactions to keep
                         EXAMPLE: take_first_fraction=0.75 -> the last 0.25 of all user
                                  transactions will be chosen as user target distribution
                                  and therefore will be cutted off
    '''

    table = pq.read_table(file_name_in)
    df = table.to_pandas()
    np_data = df.to_numpy()


    tr_amounts_by_value = {}
    for i in range(len(df)):
        tr_types_ = np_data[i][TR_TYPES_COL]
        tr_amounts_ = transform_inv(np_data[i][TR_AMOUNTS_COL])
        for ix_, j in enumerate(tr_types_):
            if j in tr_amounts_by_value:
                tr_amounts_by_value[j] += tr_amounts_[ix_]
            else:
                tr_amounts_by_value[j] = tr_amounts_[ix_]

    amounts_by_val_sorted = sorted(tr_amounts_by_value.items(),
                                   key = lambda x: abs(x[1]), reverse = True)

    negative_items = []
    positive_items = []
    for pair in amounts_by_val_sorted:
        if pair[1] >= 0:
            positive_items += [pair[0]]
        else:
            negative_items += [pair[0]]

    set_neg_types = set(negative_items)
    set_pos_types = set(positive_items)
    set_top_neg_types = set(negative_items[:top_THR])
    set_top_pos_types = set(positive_items[:top_THR])


    sums_of_negative_target = [0 for _ in range(len(df))]
    sums_of_positive_target = [0 for _ in range(len(df))]

    neg_distribution = [[] for _ in range(len(df))]
    pos_distribution = [[] for _ in range(len(df))]

    for i in range(len(df)):
        thr_target_ix = int(len(np_data[i][TR_TYPES_COL]) * take_first_fraction)

        tr_type_target = np_data[i][TR_TYPES_COL][thr_target_ix:]
        amount_target = transform_inv(np_data[i][TR_AMOUNTS_COL][thr_target_ix:])

        neg_tr_amounts_target = {}
        others_neg_tr_amounts_target = 0
        pos_tr_amounts_target = {}
        others_pos_tr_amounts_target = 0

        for ixx, el in enumerate(tr_type_target):
            if el in set_neg_types:
                sums_of_negative_target[i] += amount_target[ixx]
            else:
                sums_of_positive_target[i] += amount_target[ixx]

            if el in set_top_neg_types:
                neg_tr_amounts_target[el] = neg_tr_amounts_target.get(el, 0) + amount_target[ixx]
            elif el in set_top_pos_types:
                pos_tr_amounts_target[el] = pos_tr_amounts_target.get(el, 0) + amount_target[ixx]
            elif el in set_neg_types:
                others_neg_tr_amounts_target += amount_target[ixx]
            elif el in set_pos_types:
                others_pos_tr_amounts_target += amount_target[ixx]
            else:
                assert False, "Should not be!"

        for j in set_top_neg_types:
            if j in neg_tr_amounts_target:
                p_neg = neg_tr_amounts_target[j] / sums_of_negative_target[i]
            else:
                p_neg = 0.
            neg_distribution[i] += [p_neg]
        if sums_of_negative_target[i] != 0:
            neg_distribution[i] += [others_neg_tr_amounts_target / sums_of_negative_target[i]]
        else:
            neg_distribution[i] += [0.]

        for j in set_top_pos_types:
            if j in pos_tr_amounts_target:
                p_pos = pos_tr_amounts_target[j] / sums_of_positive_target[i]
            else:
                p_pos = 0.
            pos_distribution[i] += [p_pos]
        if sums_of_positive_target[i] != 0:
            pos_distribution[i] += [others_pos_tr_amounts_target / sums_of_positive_target[i]]
        else:
            pos_distribution[i] += [0.]


    data_to_write = list(zip(df['customer_id'], zip(sums_of_negative_target,
                                                    neg_distribution,
                                                    sums_of_positive_target,
                                                    pos_distribution)))
    output_data_df = pd.DataFrame(data=data_to_write, columns=['customer_id', 'gender'])
    output_data_df.to_csv(filename_out, index=False)


    return sums_of_negative_target, neg_distribution, sums_of_positive_target, pos_distribution


if __name__ == '__main__':
    create_new_targets_on_gender_train_csv('data/train_trx.parquet',
                                           'data/gender_train_distribution_target.csv')