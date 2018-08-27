import numpy as np
import pandas as pd
import gc

def dummy_encoding(df):
    prev_cat_features = [
        f_ for f_ in df.columns if df[f_].dtype == 'object'
    ]
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        dummy_feats = pd.get_dummies(df[f_], prefix=f_).astype(np.uint8)
        prev_dum = pd.concat([prev_dum, dummy_feats], axis=1)

    df = pd.concat([df, prev_dum], axis=1)
    del prev_dum
    gc.collect()
    return df

def count_encoding(df, column='NAME_CONTRACT_STATUS'):
    cnt_prev = (
        df[['SK_ID_CURR', 'SK_ID_PREV', column]]
        .groupby(['SK_ID_CURR', column])
        .count().unstack().fillna(0)
         )
    cnt_prev.columns = cnt_prev.columns.get_level_values(1)
    sum_cnt_prev = cnt_prev.sum(axis=1)
    for c_ in cnt_prev.columns:
        cnt_prev['X_' + c_ + '_Percent'] = cnt_prev[c_] / sum_cnt_prev
    cnt_prev.fillna(0)
    return cnt_prev

def engineering(prev):
    prev = prev[prev['NFLAG_LAST_APPL_IN_DAY'] == 1]
    prev = prev[prev['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y']

    # prev['IS_ACTIVE'] = ((prev.NAME_CONTRACT_STATUS == 'Approved') & (prev.DAYS_TERMINATION > 0)).astype(int)

    # A lot of the continuous days variables have integers as missing value indicators.
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)

    prev['IS_TERMINATED'] = (prev['DAYS_TERMINATION'] < 0).astype(int)
    prev['IS_EARLY_END'] = (prev['DAYS_LAST_DUE_1ST_VERSION'] > prev['DAYS_LAST_DUE'] ).astype(int)
    # prev['IS_DEFERED_END'] = (prev['DAYS_LAST_DUE_1ST_VERSION'] < prev['DAYS_LAST_DUE']).astype(int)

    # Agg
    prev_agg = prev.groupby('SK_ID_CURR').size().to_frame()
    prev_agg.columns = ['X_PREV_COUNT']
    prev_agg.reset_index()

    # Approved
    approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'].copy()
    approved.drop(['NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON'], inplace=True, axis=1)
    approved.drop(['HOUR_APPR_PROCESS_START', 'NAME_GOODS_CATEGORY', 'NAME_CASH_LOAN_PURPOSE'], inplace=True, axis=1)
    # approved['X_HOUR_APPR_PROCESS_START'] = approved['HOUR_APPR_PROCESS_START'].astype(str) # To categorical feature
    approved = dummy_encoding(approved)
    agg = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'DAYS_DECISION': ['max', 'min'],
        # ?
        'DAYS_FIRST_DRAWING': ['max', 'min'],
        # 'DAYS_FIRST_DUE': None, #['max', 'min'], # 初回の支払い日
        # 'DAYS_LAST_DUE_1ST_VERSION': ['max', 'min'], # 契約時の支払い完了予定日
        'DAYS_LAST_DUE': ['max'], # 支払い完了予定日。未来の場合が365243が入る
        'DAYS_TERMINATION': ['max'], # 支払い完了日。未来の場合が365243が入る
        # 'IS_ACTIVE': ['sum'],
        'IS_EARLY_END': ['mean', 'sum'],
        # 'IS_DEFERED_END': ['mean', 'sum'],
    }
    for c in approved.columns:
        ignore = ['SK_ID_PREV', 'SK_ID_CURR', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION']
        if c in ignore or c in agg:
            continue
        if approved[c].dtype != 'object':
            agg[c] = ['mean']

    approved = approved.groupby('SK_ID_CURR').agg(agg)
    approved.columns = pd.Index([e[0] + "_" + e[1].upper() for e in approved.columns.tolist()])

    # 差額の平均
    diff = (approved['AMT_APPLICATION_MEAN'] - approved['AMT_CREDIT_MEAN']) / approved['AMT_CREDIT_MEAN']
    approved['X_AMT_APPLICATION_DIFF_RATIO'] = diff
    del approved['AMT_APPLICATION_MEAN']

    # Refused
    refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']
    refused = refused.sort_values(['DAYS_DECISION'], ascending=False)
    refused = refused[['SK_ID_CURR', 'CODE_REJECT_REASON', 'AMT_APPLICATION']].groupby('SK_ID_CURR').nth(0)
    refused['X_REFUSED_AMT_APPLICATION'] = refused['AMT_APPLICATION']
    del refused['AMT_APPLICATION']
    refused['CODE_REJECT_REASON'] = refused['CODE_REJECT_REASON'].astype('category')

    # Last app features
    prev_sorted = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
    last_app = prev_sorted.groupby(by=['SK_ID_CURR']).last().reset_index()
    last_app_merge = pd.DataFrame({})
    last_app_merge['SK_ID_CURR'] = last_app['SK_ID_CURR']
    last_app_merge['X_PREV_WAS_APPROVED'] = (last_app['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    last_app_merge['X_PREV_WAS_REFUSED'] = (last_app['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    del last_app

    # Cnt
    cnt_name_contract_status = count_encoding(prev, 'NAME_CONTRACT_STATUS')

    # join
    prev_agg = prev_agg.merge(right=approved.reset_index(), how="left", on="SK_ID_CURR")
    prev_agg = prev_agg.merge(right=refused.reset_index(), how="left", on="SK_ID_CURR")
    prev_agg = prev_agg.merge(right=last_app_merge, how="left", on="SK_ID_CURR")
    prev_agg = prev_agg.merge(right=cnt_name_contract_status.reset_index(), how="left", on="SK_ID_CURR")

    # for day in [90, 180, 360]:
    #     last_nday_agg = {
    #         'DAYS_FIRST_DRAWING': ['mean'],
    #         'AMT_CREDIT': ['mean'],
    #         'CNT_PAYMENT': ['mean'],
    #         'RATE_DOWN_PAYMENT': ['mean'],
    #     }
    #     day_agg = prev[prev.DAYS_DECISION > -1 * day].groupby('SK_ID_CURR').agg(last_nday_agg)
    #     day_agg.columns = pd.Index([f"last_{day}_" +  e[0] + "_" + e[1].upper() for e in day_agg.columns.tolist()])
    #     prev_agg = prev_agg.merge(right=day_agg.reset_index(), how="left", on="SK_ID_CURR")

    for number in [1, 2, 3, 5]:
        last_nth_agg = {
            'DAYS_FIRST_DRAWING': ['mean'],
            'AMT_CREDIT': ['mean'],
            'CNT_PAYMENT': ['mean'],
            'RATE_DOWN_PAYMENT': ['mean'],
            'DAYS_DECISION': ['mean'],
        }
        nth_agg = prev_sorted.groupby('SK_ID_CURR').head(number).groupby('SK_ID_CURR').agg(last_nth_agg)
        nth_agg.columns = pd.Index([f"last_{number}th_" +  e[0] + "_" + e[1].upper() for e in nth_agg.columns.tolist()])
        prev_agg = prev_agg.merge(right=nth_agg.reset_index(), how="left", on="SK_ID_CURR")

    prev_agg = prev_agg.set_index('SK_ID_CURR')
    return prev_agg