import gc
import json
import os
import re

from lightgbm.plotting import plot_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import src.feats as feats
import src.utils as utils

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

def load_prev():
    prev = utils.read_csv('./input/previous_application.csv')

    prev_cnt = (
        prev[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS']]
        .groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])
        .count()
        .unstack()
        .fillna(0)
        )
    prev_cnt.columns = prev_cnt.columns.get_level_values(1)

    # TODO: CODE_REJECT_REASON

    # prev_refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']
    prev = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved']
    del prev['NAME_CONTRACT_STATUS']
    del prev['CODE_REJECT_REASON']

    prev['X_HOUR_APPR_PROCESS_START'] = prev['HOUR_APPR_PROCESS_START'].astype(str)
    del prev['HOUR_APPR_PROCESS_START']

    prev_cat_features = [
        f_ for f_ in prev.columns if prev[f_].dtype == 'object'
    ]
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)

    prev = pd.concat([prev, prev_dum], axis=1)
    del prev_dum
    gc.collect()

    avg_prev = prev.groupby('SK_ID_CURR').mean()
    del avg_prev['SK_ID_PREV']

    avg_prev = avg_prev.reset_index().merge(right=prev_cnt.reset_index(), how="left", on="SK_ID_CURR")
    avg_prev = avg_prev.set_index('SK_ID_CURR')
    return avg_prev

def load_buro():
    buro = utils.read_csv('./input/bureau.csv')
    buro_cat_features = [
        f_ for f_ in buro.columns if buro[f_].dtype == 'object'
    ]
    buro_dum = pd.DataFrame()
    for f_ in buro_cat_features:
        buro_dum = pd.concat([buro_dum, pd.get_dummies(buro[f_], prefix=f_).astype(np.uint8)], axis=1)

    buro = pd.concat([buro, buro_dum], axis=1)

    # for f_ in buro_cat_features:
    #     buro[f_], _ = pd.factorize(buro[f_])
    avg_buro = buro.groupby('SK_ID_CURR').mean()
    avg_buro['X_BURO_COUNT'] = buro[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
    del avg_buro['SK_ID_BUREAU']

    return avg_buro

def load_pos():
    pos = utils.read_csv('./input/POS_CASH_balance.csv')
    pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_pos = pos.groupby('SK_ID_CURR').mean()

    del pos, nb_prevs
    gc.collect()

    return avg_pos

def load_cc_bal():
    cc_bal = utils.read_csv('./input/credit_card_balance.csv')
    cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'])], axis=1)

    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()

    del cc_bal, nb_prevs
    gc.collect()

    return avg_cc_bal

def load_inst():
    inst = utils.read_csv('./input/installments_payments.csv')
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    avg_inst = inst.groupby('SK_ID_CURR').mean()
    del inst
    gc.collect()

    return avg_inst

def load_train_test():
    train = utils.read_csv('./input/application_train.csv')
    test = utils.read_csv('./input/application_test.csv')
    y = train['TARGET']
    del train['TARGET']

    feats.app_features(train)
    feats.app_features(test)

    categorical_feats = [
        f for f in train.columns if train[f].dtype == 'object'
    ]
    for f in categorical_feats:
        train[f], indexer = pd.factorize(train[f])
        test[f] = indexer.get_indexer(test[f])

    return (train, test, y)

def load_data(debug=False):
    gc.enable()

    train, test, y = load_train_test()
    prev = load_prev()
    buro = load_buro()
    pos = load_pos()
    cc_bal = load_cc_bal()
    inst = load_inst()

    prev.columns = ['prev_{}'.format(c) for c in prev.columns]
    buro.columns = ['buro_{}'.format(c) for c in buro.columns]
    inst.columns = ['inst_{}'.format(c) for c in inst.columns]
    pos.columns = ['pos_{}'.format(c) for c in pos.columns]
    cc_bal.columns = ['cc_bal_{}'.format(c) for c in cc_bal.columns]

    train = train.merge(right=prev.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=buro.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=inst.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=pos.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=cc_bal.reset_index(), how='left', on='SK_ID_CURR')

    test = test.merge(right=prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=buro.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=inst.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=cc_bal.reset_index(), how='left', on='SK_ID_CURR')

    for c_ in prev_cnt.columns:
        train[c_] = train[c_].fillna(0)
        test[c_] = test[c_].fillna(0)

    if debug:
        return locals()
    return (train, test, y)

if __name__ == '__main__':
    train, test, y = load_data()

    # Features
    excluded_feats = ['SK_ID_CURR']
    excluded_feats = sum(list(map(lambda c: [c, f"{c}_x", f"{c}_y"], excluded_feats)), [])
    features = [f_ for f_ in train.columns if f_ not in excluded_feats]

    # lgbm_train_kfold(
    #     train=train,
    #     y=y,
    #     test=test,
    #     features=features,
    # )

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
        clf=clf
    )

    reg = re.compile(r"(.*_)?X_")
    handmade_features = [x for x in features if reg.match(x)]
    for f in handmade_features:
        print('{}: {}'.format(f, clf.feature_importances_[features.index(f)]))

    # plot_importance(clf, max_num_features=100, figsize=(20, 20))