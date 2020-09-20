import pickle
import pandas as pd

path = '/mnt/data/kruzhilov/pytorch-lifestream/experiments/scenario_spend_only_rur/'
with open(path + 'data/mles_embeddings.pickle', 'rb') as f:
        data_test = pickle.load(f)
test_id = data_test['client_id']
print(data_test['client_id'])

with open('data/mles_embeddings.pickle', 'rb') as f:
        data_train = pickle.load(f)
print(data_train['client_id'])
train_id = data_train['client_id']

intersection = set(test_id) - set(train_id)
print('intersection',len(intersection))


train_target = pd.read_csv('data/train_target.csv')
print(train_target.shape)
train_target = pd.read_csv(path+'data/train_target.csv')
print(train_target.shape)

