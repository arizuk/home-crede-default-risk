import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
from lightgbm import Dataset
import gc

train = pd.read_csv('./input/application_train.csv')

categorical_feats = [
    f for f in train.columns if train[f].dtype == 'object'
]
y = train['TARGET']
del train['TARGET']

for f in categorical_feats:
    train[f], indexer = pd.factorize(train[f])

trn_x, val_x, trn_y, val_y = train_test_split(train, y,  test_size=0.2, random_state=42)

params = {
    'n_estimators': 4000,
    'learning_rate': 0.01,
    'num_leaves': 63,
    'colsample_bytree': .8,
    'subsample': .8,
    'subsample_freq': 5,
    'max_depth': 5,
    'reg_alpha': .001,
    'reg_lambda': .1,
    'min_split_gain': .01,
    'device': "gpu",
    "verbose": -1,
}
clf = LGBMClassifier(**params)

eval_set = [(trn_x, trn_y), (val_x, val_y)]

# print('name:{}'.format(','.join(categorical_feats)))

clf.fit(
    trn_x, trn_y, eval_set=eval_set, eval_metric='auc', verbose=250, early_stopping_rounds=150,
    categorical_feature=categorical_feats
    )

# auc = roc_auc_score(val_y, val_preds)
# print('AUC : %.6f' % auc)