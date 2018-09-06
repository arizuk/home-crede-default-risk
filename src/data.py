import gc
import pandas as pd
import numpy as np
import functools
import pickle
from tqdm import tqdm

from src import utils
from src import feats
from src.utils import logit
from src.utils import pd_df_cache

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

@pd_df_cache('prev')
def load_prev():
    prev = utils.read_csv('./input/previous_application.csv')
    score = pd.read_csv('./features/previous_score.csv')
    prev = prev.merge(right=score, how="left", on="SK_ID_PREV")
    return feats.previous.engineering(prev)

@pd_df_cache('inst')
def load_inst():
    inst = utils.read_csv('./input/installments_payments.csv')

    def agg(df):
        df['X_DPD'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
        df['X_OVERDUE'] = df['DAYS_INSTALMENT'] < df['DAYS_ENTRY_PAYMENT']
        df['X_OVER_PAYMENT_RATIO'] = (df['AMT_PAYMENT'] - df['AMT_INSTALMENT']) / df['AMT_PAYMENT']

        # NUM_INSTALMENT_VERSION
        inst_version = df[['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION']].groupby(['SK_ID_CURR', 'SK_ID_PREV']).max()
        inst_version = inst_version.reset_index().groupby('SK_ID_CURR').mean()
        df['X_NUM_INSTALMENT_VERSION'] = df['SK_ID_CURR'].map(inst_version['NUM_INSTALMENT_VERSION'])
        del df['NUM_INSTALMENT_VERSION']
        del inst_version

        # 支払い回数
        nb_prevs = df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
        df['X_CNT_INSTALLMENT'] = df['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

        # 遅延回数
        nb_overdues = df[df['X_OVERDUE']][['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
        df['X_CNT_OVERDUE'] = df['SK_ID_CURR'].map(nb_overdues['SK_ID_PREV'])
        df['X_CNT_OVERDUE'] = df['X_CNT_OVERDUE'].fillna(0)
        df['X_OVERDUE_RATE'] = df['X_CNT_OVERDUE'] / df['X_CNT_INSTALLMENT']

        avg_inst = df.groupby('SK_ID_CURR').mean()
        del df
        gc.collect()

        for c in ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'X_OVERDUE']:
            del avg_inst[c]
        return avg_inst

    inst = inst.sort_values(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])
    head_inst = inst.groupby('SK_ID_PREV').head(3).copy()

    avg_inst = agg(inst)
    avg_head_inst = agg(head_inst)
    avg_head_inst.columns = ['head_{}'.format(c) for c in avg_head_inst.columns]

    avg_inst = avg_inst.reset_index()
    avg_inst = avg_inst.merge(right=avg_head_inst.reset_index(), how="left", on="SK_ID_CURR")
    avg_inst = avg_inst.set_index('SK_ID_CURR')
    return avg_inst

@pd_df_cache('buro_balance')
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

@pd_df_cache('buro')
def load_buro():
    buro = utils.read_csv('./input/bureau.csv')
    buro.loc[buro['DAYS_CREDIT_ENDDATE'] < -40000, 'DAYS_CREDIT_ENDDATE'] = np.nan
    buro.loc[buro['DAYS_CREDIT_UPDATE'] < -40000, 'DAYS_CREDIT_UPDATE'] = np.nan
    buro.loc[buro['DAYS_ENDDATE_FACT'] < -40000, 'DAYS_ENDDATE_FACT'] = np.nan

    score = pd.read_csv('./features/bureau_score.csv')
    score = score.merge(right=buro[['SK_ID_BUREAU', 'SK_ID_CURR']], how="left", on="SK_ID_BUREAU")
    score_agg = score.groupby('SK_ID_CURR').SCORE.agg(['mean', 'max', 'min', 'std'])
    score_agg.columns = pd.Index(["SCORE_" + e.upper() for e in score_agg.columns.tolist()])
    del score

    # cnt
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

    # dummy encoding
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
    avg_buro = avg_buro.merge(right=score_agg.reset_index(), how="left", on="SK_ID_CURR")
    avg_buro = avg_buro.set_index('SK_ID_CURR')

    del cnt_buro, sum_buro, buro, closed_buro
    del avg_buro['SK_ID_BUREAU']
    return avg_buro

@pd_df_cache('pos')
def load_pos():
    pos = utils.read_csv('./input/POS_CASH_balance.csv')

    pos = pos.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'])
    prev_pos_gr = pos.groupby('SK_ID_PREV')

    prev_last = prev_pos_gr.last()

    # MONTHS_BALANCE == 1 and CNT_INSTALMENT_FUTURE > 1 なら支払いが残っているとみなす
    prev_last['IS_ACTIVE'] = ((prev_last.MONTHS_BALANCE == -1) & (prev_last.CNT_INSTALMENT_FUTURE > 1)).astype(int)
    prev_last_agg = {
        'IS_ACTIVE': ['sum'],
    }
    curr_prev_last = prev_last.groupby('SK_ID_CURR').agg(prev_last_agg)
    curr_prev_last.columns = pd.Index(['LAST_' + e[0] + "_" + e[1].upper() for e in curr_prev_last.columns.tolist()])

    active = prev_last[prev_last.IS_ACTIVE == 1]
    active_agg = {
        # activeな残り支払い回数
        'CNT_INSTALMENT_FUTURE': ['sum'],
    }
    curr_active = active.groupby('SK_ID_CURR').agg(active_agg)
    curr_active.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in curr_active.columns.tolist()])

    def agg(df):
        df['POS_OVERDUE'] = (df.SK_DPD > 0).astype(int)
        df['POS_OVERDUE_DEF'] = (df.SK_DPD_DEF > 0).astype(int)
        pos_aggs = {
            # 支払いがあった期間
            'MONTHS_BALANCE': ['max', 'min', 'mean'],
            'SK_DPD': ['mean', 'std'],
            'SK_DPD_DEF': ['mean', 'std'],
            'POS_OVERDUE': ['sum', 'mean'],
            'POS_OVERDUE_DEF': ['sum', 'mean'],
        }
        pos_agg = df.groupby('SK_ID_CURR').agg(pos_aggs)
        pos_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
        pos_agg['X_POS_COUNT'] = df.groupby('SK_ID_CURR').size()
        # pos_agg['SK_DPD_kurtosis'] = df.groupby('SK_ID_CURR').SK_DPD.apply(pd.Series.kurt)
        # pos_agg['SK_DPD_DEF_kurtosis'] = df.groupby('SK_ID_CURR').SK_DPD_DEF.apply(pd.Series.kurt)
        # pos_agg['X_POS_OVERDUE_RATIO'] = pos['POS_OVERDUE'] / pos_agg['X_POS_COUNT']
        # pos_agg['X_POS_OVERDUE_DEF_RATIO'] = pos['POS_OVERDUE_DEF'] / pos_agg['X_POS_COUNT']
        return pos_agg

    pos_agg = agg(pos)

    # head_pos = prev_pos_gr.head(3).copy()
    # head_pos_agg = agg(head_pos)
    # head_pos_agg.columns = ['head_{}'.format(c) for c in head_pos_agg.columns]

    # pos_agg = pos_agg.merge(right=head_pos_agg.reset_index(), how="left", on="SK_ID_CURR")
    pos_agg = pos_agg.merge(right=curr_prev_last.reset_index(), how="left", on="SK_ID_CURR")
    pos_agg = pos_agg.merge(right=curr_active.reset_index(), how="left", on="SK_ID_CURR")
    pos_agg = pos_agg.set_index('SK_ID_CURR')

    del pos, curr_prev_last, curr_active
    gc.collect()
    return pos_agg

@pd_df_cache('cc_bal')
def load_cc_bal():
    cc_bal = utils.read_csv('./input/credit_card_balance.csv')

    cc_bal.loc[cc_bal['AMT_DRAWINGS_ATM_CURRENT'] < 0, 'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
    cc_bal.loc[cc_bal['AMT_DRAWINGS_CURRENT'] < 0, 'AMT_DRAWINGS_CURRENT'] = np.nan

    cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'])], axis=1)

    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    cc_bal['weights'] = -1 / cc_bal['MONTHS_BALANCE']
    # cc_bal['weights'] = np.log(1 + -1 / cc_bal['MONTHS_BALANCE'])
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

    for df in [train, test]:
        # feats.combine_categories(df)
        feats.app_features(df)
    # feats.age_income_ratio(train, test)

    # 変数の枠だけ作っておく
    categorical_feats = [
        f for f in train.columns if train[f].dtype == 'object'
    ]
    for f in categorical_feats:
        train["X_{}_COUNT".format(f)] = np.zeros(train.shape[0])
        test["X_{}_COUNT".format(f)] = np.zeros(test.shape[0])
        # feats.merge_minor_category(train, test, f)
        feats.income_median(train, test, f)

    y = train['TARGET']
    del train['TARGET']

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

    def dmerge(df):
        df = df.merge(right=prev.reset_index(), how='left', on='SK_ID_CURR')
        df = df.merge(right=buro.reset_index(), how='left', on='SK_ID_CURR')
        df = df.merge(right=inst.reset_index(), how='left', on='SK_ID_CURR')
        df = df.merge(right=pos.reset_index(), how='left', on='SK_ID_CURR')
        df = df.merge(right=cc_bal.reset_index(), how='left', on='SK_ID_CURR')
        feats.combined_features(df)
        return df

    train = dmerge(train)
    test = dmerge(test)

    # TODO: fillna必要かも
    # for c_ in cnt_prev.columns:
    #     train[c_] = train[c_].fillna(0)
    #     test[c_] = test[c_].fillna(0)

    if debug:
        return locals()
    return (train, test, y)
