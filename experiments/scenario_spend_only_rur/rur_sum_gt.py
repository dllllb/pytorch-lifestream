import pandas as pd
import numpy as np

data = pd.read_csv("train_target.csv")
print(data.head())
        
rur_sum = data.iloc[:,2:54].to_numpy()*data.iloc[:, 54:].to_numpy()
transac_num = np.tile(data.iloc[:,1].to_numpy(), (52,1)).transpose()
rur_sum = transac_num*rur_sum #rur for each type of transaction
sum_trans = np.sum(rur_sum, axis=1)#total cost for a persor in rur
spend_persent = rur_sum/np.tile(sum_trans, (52,1)).transpose()
col_names = ['p_trans'+str(i) for i in range(52)]
df_ground_truth = pd.DataFrame(spend_persent, columns=col_names)
df_ground_truth.insert(0, "client_id", data['client_id'])
print(df_ground_truth)
df_ground_truth.to_csv('./data/train_target.csv', index=False)
