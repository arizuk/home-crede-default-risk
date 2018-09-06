from src import data
from src import utils

from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import gc
from lightgbm import LGBMClassifier

def label_encoding(df):
    obj_cols = [c for c in df.columns if df[c].dtype=='O']
    for c in obj_cols:
        df[c] = pd.factorize(df[c], na_sentinel=-1)[0]
    df[obj_cols].replace(-1, np.nan, inplace=True)
    return df

if __name__ == '__main__':
    target = utils.read_csv('./input/application_train.csv')
    target = target[['SK_ID_CURR', 'TARGET']]

    bureau = utils.read_csv('./input/bureau.csv')
    bureau = bureau.merge(how="left", right=target, on="SK_ID_CURR")

    label_encoding(bureau)

    test =bureau[bureau.TARGET.isnull()]
    test_idx =bureau[bureau.TARGET.isnull()].index.values

    train =bureau[bureau.TARGET.notnull()]
    train_idx =bureau[bureau.TARGET.notnull()].index.values
    y = train.TARGET

    sk_id_bureau =bureau.SK_ID_BUREAU
    tr_sk_id_curr = train.SK_ID_CURR
    train.drop(['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU'], inplace=True, axis=1)
    test.drop(['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU'], inplace=True, axis=1)
    print(train.columns.values)

    oof_preds = np.zeros(train.shape[0])
    test_preds = np.zeros(test.shape[0])

    folds = GroupKFold(n_splits=5)
    #folds = KFold(n_splits=5, shuffle=True, random_state=42)
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, y, groups=tr_sk_id_curr)):
    #for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, y)):
        trn_x, trn_y = train.iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = train.iloc[val_idx], y.iloc[val_idx]

        params = {
            "n_estimators": 50000,
            "learning_rate": 0.02,
            "num_leaves": 63,
            "colsample_bytree": 0.4,
            "subsample": 0.8,
            "subsample_freq": 5,
            "max_depth": 5,
            "reg_alpha": 0.001,
            "reg_lambda": 0.9,
            "min_split_gain": 0.01,
            "verbose": "-1",
            # "device": "gpu"
        }

        clf = LGBMClassifier(**params)
        with utils.timeit():
            clf.fit(trn_x, trn_y,
                    eval_set= [(trn_x, trn_y), (val_x, val_y)],
                    eval_metric='auc', verbose=250, early_stopping_rounds=200
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        test_preds += clf.predict_proba(test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        val_auc = roc_auc_score(val_y, oof_preds[val_idx])
        print('Fold %2d AUC : %.6f' % (n_fold + 1, val_auc))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    auc = roc_auc_score(y, oof_preds)
    print('Full AUC score %.6f' % auc)

    score = np.zeros(bureau.shape[0])
    score[train_idx] = oof_preds
    score[test_idx] = test_preds

    df = pd.DataFrame({
        'SK_ID_BUREAU': sk_id_bureau,
        'SCORE': score,
    })
    df.to_csv('./features/bureau_score.csv', index=False)


