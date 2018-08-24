import argparse
import gc
import json
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from lightgbm.plotting import plot_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src import feats
from src import utils
from src import data

def lgbm_train_kfold(train, y, test, features):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trn_preds = np.zeros(train.shape[0])
    oof_preds = np.zeros(train.shape[0])
    test_preds = np.zeros(test.shape[0])

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
        'verbose': -1,
    }

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, y)):
        trn_x, trn_y = train[features].iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = train[features].iloc[val_idx], y.iloc[val_idx]

        clf = LGBMClassifier(**params)
        with utils.timeit():
            clf.fit(trn_x, trn_y,
                    eval_set= [(trn_x, trn_y), (val_x, val_y)],
                    eval_metric='auc', verbose=250, early_stopping_rounds=150
                )

        trn_preds[trn_idx] = clf.predict_proba(trn_x, num_iteration=clf.best_iteration_)[:, 1]
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        test_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        val_auc = roc_auc_score(val_y, oof_preds[val_idx])
        print('Fold %2d AUC : %.6f' % (n_fold + 1, val_auc))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    clf=None
    auc = roc_auc_score(y, oof_preds)
    print('Full AUC score %.6f' % auc)
    trn_auc = roc_auc_score(y, trn_preds)

    config = {
        'model': 'lgbm',
        'params': params,
        'trn_auc': trn_auc,
        'auc': auc,
    }

    return {
        'clf': clf,
        'config': config,
        'test_preds': test_preds,
    }

def lgbm_train(train, y, test, features):
    test_preds = np.zeros(test.shape[0])
    trn_aucs = []
    aucs = []
    random_states = [1, 42]

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
        'verbose': -1,
    }

    for i in range(0, 2):
        trn_x, val_x, trn_y, val_y = train_test_split(train[features], y,  test_size=0.2, random_state=random_states[i])

        clf = LGBMClassifier(**params)

        eval_set = [(trn_x, trn_y), (val_x, val_y)]
        with utils.timeit():
            clf.fit(trn_x, trn_y, eval_set=eval_set, eval_metric='auc', verbose=250, early_stopping_rounds=150)

        trn_preds = clf.predict_proba(trn_x, num_iteration=clf.best_iteration_)[:, 1]
        val_preds = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        test_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1]

        trn_auc = roc_auc_score(trn_y, trn_preds)
        auc = roc_auc_score(val_y, val_preds)
        print('Iter %d AUC : %.6f' % (i+1, auc))
        trn_aucs.append(trn_auc)
        aucs.append(auc)

        del trn_x, val_x, trn_y, val_y
        gc.collect()

    test_preds = test_preds/2
    trn_auc = np.array(trn_aucs).mean()
    auc = np.array(aucs).mean()
    print('AUC train_auc %.6f val_auc %.6f' % (trn_auc, auc))

    config = {
        'model': 'lgbm',
        'params': params,
        'trn_auc': trn_auc,
        'auc': auc,
    }

    return {
        'clf': clf,
        'config': config,
        'test_preds': test_preds,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold', action='store_true')
    args = parser.parse_args()

    train, test, y = data.load_data()

    # Features
    excluded_feats = ['SK_ID_CURR']
    excluded_feats = sum(list(map(lambda c: [c, f"{c}_x", f"{c}_y"], excluded_feats)), [])
    features = [f_ for f_ in train.columns if f_ not in excluded_feats]

    from hyperopt import hp, tpe
    from hyperopt.fmin import fmin

    def objective(params):
        trn_x, val_x, trn_y, val_y = train_test_split(train[features], y,  test_size=0.2, random_state=1)

        params = {
            'n_estimators': 4000,
            'learning_rate': 0.01,
            # 'num_leaves': int(params['num_leaves']),
            'num_leaves': 63,
            'colsample_bytree': params['colsample_bytree'],
            'subsample': params['subsample'],
            'subsample_freq': 5,
            'max_depth': 5,
            'reg_alpha': .001,
            'reg_lambda': params['reg_lambda'],
            'min_split_gain': .01,
            'device': "gpu",
            'verbose': -1,
        }

        clf = LGBMClassifier(**params)
        eval_set = [(trn_x, trn_y), (val_x, val_y)]
        clf.fit(trn_x, trn_y, eval_set=eval_set, eval_metric='auc', verbose=250, early_stopping_rounds=150)

        val_preds = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        auc = roc_auc_score(val_y, val_preds)
        print("AUC: {:.3f} params {}".format(auc, params), flush=True)
        return auc * -1

    space = {
        # 'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'subsample': hp.uniform('subsample', 0.3, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.1, 1.0),
    }

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50)
    print(best)