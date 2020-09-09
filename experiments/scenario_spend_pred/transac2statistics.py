import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

file_path = "/home/ikruzhilov/pytorch-lifestream/experiments/scenario_spend_pred/data/transactions_train.csv"
data = pd.read_csv(file_path)
#print('example of input data:', data.head(20))
#pd.set_option('display.max_rows', None)
small_groups = data.small_group.value_counts()[data.small_group.value_counts()>45000]
#list of frequent type of transactions
valid_transaction_indexes = small_groups.index
inverted_indexes = np.zeros(shape=[valid_transaction_indexes.max()+1])
#print(inverted_indexes, valid_transaction_indexes.max())
print(valid_transaction_indexes[0])
for idx in range(valid_transaction_indexes.shape[0]):
    inverted_indexes[valid_transaction_indexes.values[idx]] = idx
inverted_indexes = inverted_indexes.astype(int)
print('inverted valid indexes:', inverted_indexes)    
valid_statistics = data[data['small_group'].isin(valid_transaction_indexes)]
non_valid_statistics = data[~data['small_group'].isin(valid_transaction_indexes)]
val_sum = valid_statistics['amount_rur'].sum()
val_count = valid_statistics['small_group'].count()
nval_sum = non_valid_statistics['amount_rur'].sum()
nval_count = non_valid_statistics['small_group'].count()
val_sum_per = val_sum/(val_sum+nval_sum)
val_count_per = val_count/(val_count+nval_count)
print('valid transactions by sum:', 100*val_sum_per, '%')
print('valid transactions by count:', 100*val_count_per, '%')
print('share of groups left:',100*valid_transaction_indexes.shape[0]/ data.small_group.value_counts().shape[0],'%')
#val_count_per = val_count/(val_count+nval_count), 'count:', valid_statistics['amount_rur'].count()
non_valid_statistics = data[~data['small_group'].isin(valid_transaction_indexes)]
#print('valid transaction indexes', valid_transaction_indexes)
#print('transactions by total number in dataset:',small_groups)
client_ids = data.client_id.unique()
#translates dates to monthes
data['trans_date'] = (data['trans_date'] - data['trans_date'].min()) // 328

#transactions for the client with a certain id
i = 0
ids_ground_truth = np.zeros(shape=[client_ids.shape[0],106])
for id in client_ids:
    id_data = data[data['client_id']==id]
    split_day = id_data.groupby('trans_date')
    month_transac = dict(tuple(split_day))[0][['small_group','amount_rur']]
    #print('transaction types and their values:', month_transac)
    transactions_grouped = month_transac.groupby(['small_group']).agg(['count', 'sum'])
    not_valid_transactions = transactions_grouped[~transactions_grouped.index.isin(valid_transaction_indexes)]
    counts_non_valid = not_valid_transactions.iloc[:,0].sum()
    rur_non_valid = not_valid_transactions.iloc[:,1].sum()
    transactions_grouped = transactions_grouped[transactions_grouped.index.isin(valid_transaction_indexes)]
    last_row = transactions_grouped.iloc[0,:]
    #print(last_row)
    last_row.iloc[0] = counts_non_valid
    last_row.iloc[1] = rur_non_valid
    #print(transactions_grouped)
    transactions_grouped = transactions_grouped.append(last_row ,ignore_index=False)
    #print(transactions_grouped)
    total_number_transactions = transactions_grouped.iloc[:,0].sum()
    normalized_counts = transactions_grouped.iloc[:,0]/total_number_transactions
    mean_rur = transactions_grouped.iloc[:,1]/transactions_grouped.iloc[:,0]
    mean_norm_counts = pd.DataFrame(index=valid_transaction_indexes ,columns = ['norm_counts', 'mean_rur'])
    mean_norm_counts = mean_norm_counts.fillna(0)
    mean_norm_counts['mean_rur'].iloc[inverted_indexes[normalized_counts.index]] = mean_rur
    mean_norm_counts['norm_counts'].iloc[inverted_indexes[normalized_counts.index]] = normalized_counts
    last_row = mean_norm_counts.iloc[0,:]
    last_row.iloc[0] = 1 - mean_norm_counts['norm_counts'].sum()
    last_row.iloc[1] = mean_rur.sum() - mean_norm_counts['mean_rur'].sum()
    last_row.rename('666')
    mean_norm_counts = mean_norm_counts.append(last_row, ignore_index=True)
    positive_rur = mean_norm_counts['mean_rur']>0
    mean_norm_counts['mean_rur'][positive_rur] = np.log(mean_norm_counts['mean_rur'][positive_rur])
    #print('valid transactions:', mean_norm_counts)
    #print('nonvalid transactions:')
    #print(not_valid_transactions)
    np_row = np.hstack([np.array([id, total_number_transactions]), mean_norm_counts['norm_counts'].to_numpy(), mean_norm_counts['mean_rur'].to_numpy()] )

    ids_ground_truth[i,:] = np_row
    #print(np_row.shape)
    i = i + 1
    #if i >1:
    #    break
col_names = ['client_id','transac_total'] + ['p_trans'+str(i) for i in range(52)] + ['mean_rur'+str(i) for i in range(52)] 
df_ground_truth = pd.DataFrame(ids_ground_truth, columns=col_names)
df_ground_truth = df_ground_truth.astype({'client_id':int})
df_ground_truth = df_ground_truth.astype({'client_id':int}) 
print(type(df_ground_truth['client_id'][1]) )
df_ground_truth.to_csv('train_target.csv', index=False)


