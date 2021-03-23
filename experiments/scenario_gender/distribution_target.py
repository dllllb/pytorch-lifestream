import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle


def transform(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def transform_inv(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


def load_data_csv(file_name_in):
    df = pd.read_csv(file_name_in)
    grouped = df.groupby('customer_id')
    res = {}
    res['tr_type'] = grouped['tr_type'].apply(list)
    res['amount'] = grouped['amount'].apply(list)
    df = pd.concat(res, axis = 1).reset_index()
    return df.to_numpy(), list(df.columns)


def load_data_pq(file_name_in):
    table = pq.read_table(file_name_in)
    df = table.to_pandas()
    return df.to_numpy(), list(df.columns)


def top_tr_types(np_data, tr_types_col, tr_amounts_col, f):
    tr_amounts_by_value = {}
    for i in range(len(np_data)):
        tr_types_ = np_data[i][tr_types_col]
        tr_amounts_ = f(np_data[i][tr_amounts_col])
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
    
    return negative_items, positive_items


def get_distributions(np_data, tr_amounts_col, tr_types_col=None,
                      negative_items=None, positive_items=None, top_thr=None,
                      take_first_fraction=0, f=lambda x: x):
    set_top_neg_types = set(negative_items[:top_thr])
    set_top_pos_types = set(positive_items[:top_thr])

    sums_of_negative_target = [0 for _ in range(len(np_data))]
    sums_of_positive_target = [0 for _ in range(len(np_data))]

    neg_distribution = [[] for _ in range(len(np_data))]
    pos_distribution = [[] for _ in range(len(np_data))]

    for i in range(len(np_data)):
        num = len(np_data[i][tr_amounts_col])
        thr_target_ix = int(num * take_first_fraction)
        amount_target = f(np_data[i][tr_amounts_col][thr_target_ix:])

        tr_type_target = np_data[i][tr_types_col][thr_target_ix:]
        neg_tr_amounts_target = {}
        others_neg_tr_amounts_target = 0
        pos_tr_amounts_target = {}
        others_pos_tr_amounts_target = 0

        # create value counts for each top transaction type and create neg and pos sum for id

        for ixx, el in enumerate(tr_type_target):
            if amount_target[ixx] < 0:
                sums_of_negative_target[i] += amount_target[ixx]
            else:
                sums_of_positive_target[i] += amount_target[ixx]

            if el in set_top_neg_types:
                neg_tr_amounts_target[el] = neg_tr_amounts_target.get(el, 0) + amount_target[ixx]
            elif el in set_top_pos_types:
                pos_tr_amounts_target[el] = pos_tr_amounts_target.get(el, 0) + amount_target[ixx]
            elif amount_target[ixx] < 0:
                others_neg_tr_amounts_target += amount_target[ixx]
            elif amount_target[ixx] >= 0:
                others_pos_tr_amounts_target += amount_target[ixx]
            else:
                assert False, "Should not be!"

        # collect neg and pos distribution for id
        for j in negative_items[:top_thr]:
            if j in neg_tr_amounts_target:
                p_neg = neg_tr_amounts_target[j] / sums_of_negative_target[i]
            else:
                p_neg = 0.
            neg_distribution[i] += [p_neg]
        if sums_of_negative_target[i] != 0:
            neg_distribution[i] += [others_neg_tr_amounts_target / sums_of_negative_target[i]]
        else:
            neg_distribution[i] += [0.]

        for j in positive_items[:top_thr]:
            if j in pos_tr_amounts_target:
                p_pos = pos_tr_amounts_target[j] / sums_of_positive_target[i]
            else:
                p_pos = 0.
            pos_distribution[i] += [p_pos]
        if sums_of_positive_target[i] != 0:
            pos_distribution[i] += [others_pos_tr_amounts_target / sums_of_positive_target[i]]
        else:
            pos_distribution[i] += [0.]
    
    return sums_of_negative_target, sums_of_positive_target, neg_distribution, pos_distribution


def write_target(filename_out, ids, sums_of_negative_target, sums_of_positive_target,
                 neg_distribution, pos_distribution):
    data_to_write = list(zip(ids, zip(sums_of_negative_target,
                                      neg_distribution,
                                      sums_of_positive_target,
                                      pos_distribution)))
    output_data_df = pd.DataFrame(data=data_to_write, columns=['customer_id', 'gender'])
    output_data_df.to_csv(filename_out, index=False)
    

def create_new_targets_on_gender_train_csv(file_name_in, filename_out, TR_AMOUNTS_COL=2,
                                           TR_TYPES_COL=1, top_THR=5, take_first_fraction=0.5):
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

    # load data to numpy
    np_data, columns = load_data_csv(file_name_in)
    ids = np_data[:, columns.index('customer_id')]

    # get top negative and positive types
    negative_items, positive_items = top_tr_types(np_data, TR_TYPES_COL, TR_AMOUNTS_COL, lambda x: x)
    with open('negative.pickle', 'wb') as neg:
        pickle.dump(negative_items, neg)
    with open('positive.pickle', 'wb') as pos:
        pickle.dump(positive_items, pos)
    
    # get distributions
    distr = get_distributions(np_data, TR_AMOUNTS_COL, TR_TYPES_COL,
                              negative_items, positive_items, top_THR, take_first_fraction)

    sums_of_negative_target, sums_of_positive_target = distr[0], distr[1]
    neg_distribution, pos_distribution = distr[2], distr[3]
    
    #write target
    write_target(filename_out, ids, sums_of_negative_target, sums_of_positive_target,
                 neg_distribution, pos_distribution)

    return sums_of_negative_target, neg_distribution, sums_of_positive_target, pos_distribution


if __name__ == '__main__':
    create_new_targets_on_gender_train_csv('data/transactions.csv',
                                           'data/gender_train_distribution_target.csv')
