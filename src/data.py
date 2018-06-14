import gc
import pandas as pd
import numpy as np
import functools
import pickle
from tqdm import tqdm

from src import utils
from src import feats
from src.utils import logit

tqdm.pandas()


def weighted_average(df, weight_col, by_col):
    columns = [c for c in df.columns if df[c].dtype != 'object' and c not in [weight_col, by_col]]

    wdf = pd.DataFrame({})
    wdf[by_col] = df[by_col]

    for c in columns:
        wdf[c] = df[c] * df[weight_col]
        wdf[f"{c}_notnull"] = df[weight_col] * pd.notnull(df[c])

    g = wdf.groupby(by_col).sum()

    avg = pd.DataFrame({})
    for c in columns:
        avg[c] = g[c] / g[f'{c}_notnull']
    return avg

@logit
def load_prev():
    prev = utils.read_csv('./input/previous_application.csv')
    prev = prev[prev['NFLAG_LAST_APPL_IN_DAY'] == 1]
    prev = prev[prev['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y']

    # A lot of the continuous days variables have integers as missing value indicators.
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)

    cnt_prev = (
        prev[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS']]
        .groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])
        .count().unstack().fillna(0)
         )
    cnt_prev.columns = cnt_prev.columns.get_level_values(1)
    sum_cnt_prev = cnt_prev.sum(axis=1)
    for c_ in cnt_prev.columns:
        cnt_prev['X_' + c_ + '_Percent'] = cnt_prev[c_] / sum_cnt_prev
    cnt_prev.fillna(0)

    refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']

    # Approved
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

    # Weighted average
    # def weighted_average(data):
    #     d = {}
    #     weights = -1 / data['DAYS_DECISION']
    #     for c in data.columns:
    #         if data[c].dtype == 'object':
    #             continue
    #         d[c] = np.average(data[c], weights=weights)
    #     return pd.Series(d)
    # avg_prev = prev.groupby('SK_ID_CURR').progress_apply(weighted_average)

    # prev['weights'] = -1 / prev.DAYS_DECISION
    # avg_prev = weighted_average(df=prev, weight_col='weights', by_col='SK_ID_CURR')
    avg_prev = prev.groupby('SK_ID_CURR').mean()
    del avg_prev['SK_ID_PREV']

    # max
    max_columns = [
        'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
        'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION',
    ]
    max_prev = prev[['SK_ID_CURR'] + max_columns].groupby('SK_ID_CURR').max()
    for f_ in max_columns:
        if f_ in ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT']:
            avg_prev['X_MAX_' + f_] = max_prev[f_]
        else:
            avg_prev[f_] = max_prev[f_]
    del max_prev

    # refused
    refused = refused.sort_values(['DAYS_DECISION'], ascending=False)
    refused = refused[['SK_ID_CURR', 'CODE_REJECT_REASON', 'AMT_APPLICATION']].groupby('SK_ID_CURR').nth(0)
    refused['X_REFUSED_AMT_APPLICATION'] = refused['AMT_APPLICATION']
    del refused['AMT_APPLICATION']
    refused['CODE_REJECT_REASON'] = refused['CODE_REJECT_REASON'].astype('category')

    # join
    avg_prev = avg_prev.reset_index()
    avg_prev = avg_prev.merge(right=cnt_prev.reset_index(), how="left", on="SK_ID_CURR")
    avg_prev = avg_prev.merge(right=refused.reset_index(), how="left", on="SK_ID_CURR")
    avg_prev = avg_prev.set_index('SK_ID_CURR')

    feats.prev_features(avg_prev)
    return avg_prev

@logit
def load_inst():
    inst = utils.read_csv('./input/installments_payments.csv')

    inst['X_DPD'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['X_OVERDUE'] = inst['DAYS_INSTALMENT'] < inst['DAYS_ENTRY_PAYMENT']
    inst['X_OVER_PAYMENT_RATIO'] = (inst['AMT_PAYMENT'] - inst['AMT_INSTALMENT']) / inst['AMT_PAYMENT']

    # NUM_INSTALMENT_VERSION
    inst_version = inst[['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION']].groupby(['SK_ID_CURR', 'SK_ID_PREV']).max()
    inst_version = inst_version.reset_index().groupby('SK_ID_CURR').mean()
    inst['X_NUM_INSTALMENT_VERSION'] = inst['SK_ID_CURR'].map(inst_version['NUM_INSTALMENT_VERSION'])
    del inst['NUM_INSTALMENT_VERSION']
    del inst_version

    # 支払い回数
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['X_CNT_INSTALLMENT'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    # 遅延回数
    nb_overdues = inst[inst['X_OVERDUE']][['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['X_CNT_OVERDUE'] = inst['SK_ID_CURR'].map(nb_overdues['SK_ID_PREV'])
    inst['X_CNT_OVERDUE'] = inst['X_CNT_OVERDUE'].fillna(0)
    inst['X_OVERDUE_RATE'] = inst['X_CNT_OVERDUE'] / inst['X_CNT_INSTALLMENT']

    avg_inst = inst.groupby('SK_ID_CURR').mean()
    del inst
    gc.collect()

    for c in ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'X_OVERDUE']:
        del avg_inst[c]

    return avg_inst

@logit
def load_last():
    last = pickle.load(open('./features/last_application.pkl', 'rb'))
    feats.prev_features(last)
    for f_ in [f for f in last.columns if last[f].dtype == 'object']:
        last[f_], indexer = pd.factorize(last[f_])
    del last['SK_ID_PREV']
    return last

@logit
def load_buro_balance():
    buro_bal = utils.read_csv('./input/bureau_balance.csv')
    buro_bal = buro_bal[buro_bal.STATUS != 'C']

    cnt_buro_bal = (
        buro_bal.groupby(['SK_ID_BUREAU', 'STATUS'])
        .count().unstack().fillna(0)
        )
    cnt_buro_bal.columns = cnt_buro_bal.columns.get_level_values(1)

    # days past due
    # dpd_columns = [c for c in cnt_buro_bal.columns if c in ['1', '2', '3', '4']]
    # cnt_buro_bal['DPD'] = np.zeros(cnt_buro_bal.shape[0])
    # for c in dpd_columns:
    #     cnt_buro_bal['DPD'] = cnt_buro_bal['DPD'] + cnt_buro_bal[c]
    #     del cnt_buro_bal[c]

    sum_cnt_buro = cnt_buro_bal.sum(axis=1)
    for c_ in cnt_buro_bal.columns:
        cnt_buro_bal['X_' + c_ + '_Percent'] = cnt_buro_bal[c_] / sum_cnt_buro
    cnt_buro_bal.fillna(0)
    return cnt_buro_bal

@logit
def load_buro():
    buro = utils.read_csv('./input/bureau.csv')

    cnt_buro = (
        buro[['SK_ID_CURR', 'SK_ID_BUREAU', 'CREDIT_ACTIVE']]
        .groupby(['SK_ID_CURR', 'CREDIT_ACTIVE'])
        .count().unstack().fillna(0)
        )
    cnt_buro.columns = cnt_buro.columns.get_level_values(1)
    sum_cnt_buro = cnt_buro.sum(axis=1)
    for c_ in cnt_buro.columns:
        cnt_buro['X_' + c_ + '_Percent'] = cnt_buro[c_] / sum_cnt_buro
    cnt_buro.fillna(0)

    del buro['CREDIT_CURRENCY']

    buro_dum = pd.DataFrame()
    buro_cat_features = [
        f_ for f_ in buro.columns if buro[f_].dtype == 'object'
    ]
    for f_ in buro_cat_features:
        buro_dum = pd.concat([buro_dum, pd.get_dummies(buro[f_], prefix=f_).astype(np.uint8)], axis=1)

    buro = pd.concat([buro, buro_dum], axis=1)
    # for f_ in buro_cat_features:
    #     buro[f_], _ = pd.factorize(buro[f_])

    # join buro_balance
    buro_bal = load_buro_balance()
    buro_bal = buro[['SK_ID_CURR', 'SK_ID_BUREAU']].merge(right=buro_bal.reset_index(), how="left", on="SK_ID_BUREAU")
    avg_buro_bal = buro_bal.groupby('SK_ID_CURR').mean()
    del avg_buro_bal['SK_ID_BUREAU']
    del buro_bal

    # active buro sum/avg
    active_buro = buro[buro.CREDIT_ACTIVE == 'Active']
    sum_columns = [
        'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE',
    ]
    avg_columns = [c for c in buro.columns if c not in sum_columns]
    avg_buro = active_buro[avg_columns].groupby('SK_ID_CURR').mean()
    sum_buro = active_buro[['SK_ID_CURR'] + sum_columns].groupby('SK_ID_CURR').sum()
    sum_buro['X_AMT_CREDIT_SUM_DEBT_RATIO'] = sum_buro['AMT_CREDIT_SUM'] / sum_buro['AMT_CREDIT_SUM_DEBT']
    sum_buro['X_AMT_CREDIT_SUM_OVERDUE_CREDIT_RATIO'] = sum_buro['AMT_CREDIT_SUM_OVERDUE'] / sum_buro['AMT_CREDIT_SUM']
    del active_buro

    # closed buro
    closed_buro = buro[buro.CREDIT_ACTIVE == 'Closed']
    closed_buro = closed_buro[['SK_ID_CURR', 'AMT_CREDIT_SUM', 'AMT_CREDIT_MAX_OVERDUE']].groupby('SK_ID_CURR').mean()
    closed_buro.columns = ["Closed_" + c for c in closed_buro.columns]

    avg_buro = avg_buro.reset_index()
    avg_buro = avg_buro.merge(right=cnt_buro.reset_index(), how="left", on="SK_ID_CURR")
    avg_buro = avg_buro.merge(right=sum_buro.reset_index(), how="left", on="SK_ID_CURR")
    avg_buro = avg_buro.merge(right=closed_buro.reset_index(), how="left", on="SK_ID_CURR")
    avg_buro = avg_buro.merge(right=avg_buro_bal.reset_index(), how="left", on="SK_ID_CURR")
    avg_buro = avg_buro.set_index('SK_ID_CURR')

    del cnt_buro, sum_buro, buro, closed_buro
    del avg_buro['SK_ID_BUREAU']
    return avg_buro

@logit
def load_pos():
    pos = utils.read_csv('./input/POS_CASH_balance.csv')
    pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_pos = pos.groupby('SK_ID_CURR').mean()

    del pos, nb_prevs
    gc.collect()

    del avg_pos['SK_ID_PREV']
    return avg_pos

@logit
def load_cc_bal():
    cc_bal = utils.read_csv('./input/credit_card_balance.csv')
    cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'])], axis=1)

    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    cc_bal['weights'] = -1 / cc_bal['MONTHS_BALANCE']
    avg_cc_bal = weighted_average(cc_bal, 'weights', 'SK_ID_CURR')
    # avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()

    del cc_bal, nb_prevs
    gc.collect()

    del avg_cc_bal['SK_ID_PREV']
    return avg_cc_bal

@logit
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
    # last = load_last()
    buro = load_buro()
    # pos = load_pos()
    cc_bal = load_cc_bal()
    inst = load_inst()

    prev.columns = ['prev_{}'.format(c) for c in prev.columns]
    # last.columns = ['last_{}'.format(c) for c in last.columns]
    buro.columns = ['buro_{}'.format(c) for c in buro.columns]
    inst.columns = ['inst_{}'.format(c) for c in inst.columns]
    # pos.columns = ['pos_{}'.format(c) for c in pos.columns]
    cc_bal.columns = ['cc_bal_{}'.format(c) for c in cc_bal.columns]

    train = train.merge(right=prev.reset_index(), how='left', on='SK_ID_CURR')
    # train = train.merge(right=last.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=buro.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=inst.reset_index(), how='left', on='SK_ID_CURR')
    # train = train.merge(right=pos.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=cc_bal.reset_index(), how='left', on='SK_ID_CURR')

    test = test.merge(right=prev.reset_index(), how='left', on='SK_ID_CURR')
    # test = test.merge(right=last.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=buro.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=inst.reset_index(), how='left', on='SK_ID_CURR')
    # test = test.merge(right=pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=cc_bal.reset_index(), how='left', on='SK_ID_CURR')

    feats.combined_features(train)
    feats.combined_features(test)

    # TODO: fillna必要かも
    # for c_ in cnt_prev.columns:
    #     train[c_] = train[c_].fillna(0)
    #     test[c_] = test[c_].fillna(0)

    if debug:
        return locals()
    return (train, test, y)