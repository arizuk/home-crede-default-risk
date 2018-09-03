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
from src.feats import selection

CONFIG_ID=1

def get_lgbm_params():
    config_id = CONFIG_ID
    json_data=open("config/{}.json".format(config_id)).read()
    data = json.loads(json_data)
    print("[CONFIG] {}.json {}".format(config_id, data))
    return data

def lgbm_train_kfold2(train, y, test, features, random_state=42):
    categorical_feats = [ f for f in train.columns if train[f].dtype == 'object' ]
    categ_train = train[categorical_feats]
    feats.encode_categories(train=train, test=test, y=y, features=categorical_feats)

    test_preds = np.zeros(test.shape[0])

    for seed in [0, 20, 36, 42, 60]:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        trn_preds = np.zeros(train.shape[0])
        oof_preds = np.zeros(train.shape[0])

        params = get_lgbm_params()

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, y)):
            train[categorical_feats] = categ_train #restore
            trn_x, trn_y = train[features].iloc[trn_idx], y.iloc[trn_idx]
            val_x, val_y = train[features].iloc[val_idx], y.iloc[val_idx]
            feats.encode_categories(train=trn_x, test=val_x, y=trn_y, features=categorical_feats)

            clf = LGBMClassifier(**params)
            with utils.timeit():
                clf.fit(trn_x, trn_y,
                        eval_set= [(trn_x, trn_y), (val_x, val_y)],
                        eval_metric='auc', verbose=250, early_stopping_rounds=200
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

    test_preds = test_preds / 5

    return {
        'clf': clf,
        'config': config,
        'test_preds': test_preds,
    }


def lgbm_train_kfold(train, y, test, features, random_state=42):
    categorical_feats = [ f for f in train.columns if train[f].dtype == 'object' ]
    categ_train = train[categorical_feats]
    feats.encode_categories(train=train, test=test, y=y, features=categorical_feats)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    trn_preds = np.zeros(train.shape[0])
    oof_preds = np.zeros(train.shape[0])
    test_preds = np.zeros(test.shape[0])

    params = get_lgbm_params()

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, y)):
        train[categorical_feats] = categ_train #restore
        trn_x, trn_y = train[features].iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = train[features].iloc[val_idx], y.iloc[val_idx]
        feats.encode_categories(train=trn_x, test=val_x, y=trn_y, features=categorical_feats)

        clf = LGBMClassifier(**params)
        with utils.timeit():
            clf.fit(trn_x, trn_y,
                    eval_set= [(trn_x, trn_y), (val_x, val_y)],
                    eval_metric='auc', verbose=250, early_stopping_rounds=200
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
    categorical_feats = [ f for f in train.columns if train[f].dtype == 'object' ]
    categ_train = train[categorical_feats]
    feats.encode_categories(train=train, test=test, y=y, features=categorical_feats)

    test_preds = np.zeros(test.shape[0])
    trn_aucs = []
    aucs = []
    # random_states = [1, 42]
    random_states = [42, 1]

    params = get_lgbm_params()

    for i in range(0, 2):
        train[categorical_feats] = categ_train #restore
        trn_x, val_x, trn_y, val_y = train_test_split(train[features], y,  test_size=0.2, random_state=random_states[i])
        feats.encode_categories(train=trn_x, test=val_x, y=trn_y, features=categorical_feats)
        clf = LGBMClassifier(**params)
        eval_set = [(trn_x, trn_y), (val_x, val_y)]
        with utils.timeit():
            clf.fit(trn_x, trn_y, eval_set=eval_set, eval_metric='auc', verbose=250, early_stopping_rounds=200)

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
    features = selection.select_features(train.columns.values)
    print('Using features: {}'.format(len(features)), flush=True)

    if args.kfold:
        results = lgbm_train_kfold(
            train=train,
            y=y,
            test=test,
            features=features,
        )
    else:
        results = lgbm_train(
            train=train,
            y=y,
            test=test,
            features=features,
        )

    for var_ in ['config', 'test_preds', 'clf']:
        locals()[var_] = results[var_]

    utils.save_result(
        test=test,
        test_preds=test_preds,
        config=config,
        features=features,
        clf=clf,
        kfold=args.kfold
    )

    reg = re.compile(r"(.*_)?X_")
    handmade_features = [x for x in features if reg.match(x)]
    for f in handmade_features:
        pass
        # print('{}: {}'.format(f, clf.feature_importances_[features.index(f)]))

    # plot_importance(clf, max_num_features=100, figsize=(20, 20))
