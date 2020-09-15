"""
collect statistics on average sums and types of transactions
on the train dataset
"""

import pandas as pd
import numpy as np

data = pd.read_csv("data/train_target.csv")
print(data.head())
mean_q = data.iloc[:,1:53].mean()
print(np.array2string(mean_q.to_numpy(), separator=',') )
print(mean_q.shape)

#mean_rur = []
#for i in range(54,106):
#  vec_nonzero = np.where(data.iloc[:,i] != 0)
#  mean_rur.append(data.iloc[vec_nonzero[0], i].mean())

print(np.exp(np.array(mean_rur)))
print(len(mean_rur))
