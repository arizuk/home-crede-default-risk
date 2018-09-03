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

def approved_agg(prev):
    approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'].copy()
    # approved.drop(['NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON'], inplace=True, axis=1)
    # approved.drop(['HOUR_APPR_PROCESS_START', 'NAME_GOODS_CATEGORY', 'NAME_CASH_LOAN_PURPOSE'], inplace=True, axis=1)
    # approved['X_HOUR_APPR_PROCESS_START'] = approved['HOUR_APPR_PROCESS_START'].astype(str) # To categorical feature
    # approved = dummy_encoding(approved)

    # for c in approved.columns:
    #     ignore = ['SK_ID_PREV', 'SK_ID_CURR', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION']
    #     if c in ignore or c in agg:
    #         continue
    #     if approved[c].dtype != 'object':
    #         agg[c] = ['mean']

    diff = (approved['AMT_APPLICATION'] - approved['AMT_CREDIT']) / approved['AMT_CREDIT']
    approved['X_AMT_APPLICATION_DIFF_RATIO'] = diff
    aggs = {
        'AMT_ANNUITY': ['max', 'mean', 'size'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['mean'],
        'X_AMT_APPLICATION_DIFF_RATIO': ['mean'],
        #'RATE_INTEREST_PRIMARY': ['mean'],
        #'RATE_INTEREST_PRIVILEGED': ['mean'],
        #'DAYS_DECISION': ['mean'],
    }
    approved_agg = approved.groupby('SK_ID_CURR').agg(aggs)
    approved_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])

    cash_loans = approved[approved.NAME_CONTRACT_TYPE == 'Cash loans']
    aggs = {
        'AMT_ANNUITY': ['mean'],
        'AMT_APPLICATION': ['mean'],
        'AMT_CREDIT': ['mean'],
        'CNT_PAYMENT': ['mean'],
    }
    cash_loans = cash_loans.groupby('SK_ID_CURR').agg(aggs)
    cash_loans.columns = pd.Index(["cash_loans_" + e[0] + "_" + e[1].upper() for e in cash_loans.columns.tolist()])

    consumer_loans = approved[approved.NAME_CONTRACT_TYPE == 'Consumer loans']
    aggs = {
        'AMT_ANNUITY': ['mean'],
        'AMT_APPLICATION': ['mean'],
        'AMT_CREDIT': ['mean'],
        'CNT_PAYMENT': ['mean'],
        'AMT_DOWN_PAYMENT': ['mean'],
        'RATE_DOWN_PAYMENT': ['mean'],
    }
    consumer_loans = consumer_loans.groupby('SK_ID_CURR').agg(aggs)
    consumer_loans.columns = pd.Index(["consumer_loans_" + e[0] + "_" + e[1].upper() for e in consumer_loans.columns.tolist()])

    revolving_loans = approved[approved.NAME_CONTRACT_TYPE == 'Revolving loans']
    aggs = {
        'AMT_ANNUITY': ['mean'],
        'AMT_APPLICATION': ['mean'],
        'AMT_CREDIT': ['mean'],
    }
    revolving_loans = revolving_loans.groupby('SK_ID_CURR').agg(aggs)
    revolving_loans.columns = pd.Index(["revolving_loans_" + e[0] + "_" + e[1].upper() for e in revolving_loans.columns.tolist()])

    cnt_name_contract_type = count_encoding(approved, 'NAME_CONTRACT_TYPE')

    approved_agg = approved_agg.reset_index()
    approved_agg = approved_agg.merge(right=cnt_name_contract_type.reset_index(), how="left", on="SK_ID_CURR")
    approved_agg = approved_agg.merge(right=cash_loans.reset_index(), how="left", on="SK_ID_CURR")
    approved_agg = approved_agg.merge(right=consumer_loans.reset_index(), how="left", on="SK_ID_CURR")
    approved_agg = approved_agg.merge(right=revolving_loans.reset_index(), how="left", on="SK_ID_CURR")
    return approved_agg.set_index('SK_ID_CURR')

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

    # Agg
    prev_agg = prev.groupby('SK_ID_CURR').size().to_frame()
    prev_agg.columns = ['X_PREV_COUNT']
    prev_agg.reset_index()

    aggs = {
        'DAYS_DECISION': ['max', 'min', 'mean'],
        'DAYS_FIRST_DRAWING': ['max', 'min', 'mean'],
        'DAYS_FIRST_DUE': ['max', 'min', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['max', 'min', 'mean'],
        'DAYS_LAST_DUE': ['max', 'min', 'mean'],
        'DAYS_TERMINATION': ['max', 'min', 'mean'],
        'SCORE': ['mean', 'min', 'max', 'std']
    }
    all_agg = prev.groupby('SK_ID_CURR').agg(aggs)
    all_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in all_agg.columns.tolist()])
    prev_agg = prev_agg.merge(right=all_agg.reset_index(), how="left", on="SK_ID_CURR")

    # Cnt
    cnt_name_contract_status = count_encoding(prev, 'NAME_CONTRACT_STATUS')
    prev_agg = prev_agg.merge(right=cnt_name_contract_status.reset_index(), how="left", on="SK_ID_CURR")

   # Refused
    refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']
    refused = refused.sort_values(['DAYS_DECISION'], ascending=False)
    refused = refused[['SK_ID_CURR', 'CODE_REJECT_REASON', 'AMT_APPLICATION']].groupby('SK_ID_CURR').nth(0)
    refused['X_REFUSED_AMT_APPLICATION'] = refused['AMT_APPLICATION']
    del refused['AMT_APPLICATION']
    refused['CODE_REJECT_REASON'] = refused['CODE_REJECT_REASON'].astype('category')
    prev_agg = prev_agg.merge(right=refused.reset_index(), how="left", on="SK_ID_CURR")

    # Last app features
    prev_sorted = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
    last_app = prev_sorted.groupby(by=['SK_ID_CURR']).last().reset_index()
    last_app_merge = pd.DataFrame({})
    last_app_merge['SK_ID_CURR'] = last_app['SK_ID_CURR']
    last_app_merge['X_LAST_WAS_APPROVED'] = (last_app['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    last_app_merge['X_LAST_WAS_REVOLVING_LOAN'] = (last_app['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype(int)
    last_app_merge['X_LAST_NAME_CLIENT_TYPE'] = last_app.NAME_CLIENT_TYPE.astype('category')
    del last_app
    prev_agg = prev_agg.merge(right=last_app_merge, how="left", on="SK_ID_CURR")

    # Approved
    approved = approved_agg(prev)
    approved.columns = pd.Index(["approved_" + f for f in approved.columns.tolist()])
    prev_agg = prev_agg.merge(right=approved.reset_index(), how="left", on="SK_ID_CURR")

    # ## last nth approved
    # last_nth = prev_sorted[prev_sorted.NAME_CONTRACT_STATUS == "Approved"].groupby('SK_ID_CURR').head(10).copy()
    # last_nth = approved_agg(last_nth)
    # last_nth.columns = pd.Index(["approved_" + f for f in last_nth.columns.tolist()])
    # prev_agg = prev_agg.merge(right=last_nth.reset_index(), how="left", on="SK_ID_CURR")

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

    # for number in [1, 5]:
    #     last_nth_agg = {
    #         'DAYS_FIRST_DRAWING': ['mean'],
    #         'AMT_CREDIT': ['mean'],
    #         'CNT_PAYMENT': ['mean'],
    #         'RATE_DOWN_PAYMENT': ['mean'],
    #         'DAYS_DECISION': ['mean'],
    #         'IS_REFUSED': ['sum'],
    #         'IS_REVOLING': ['mean'],
    #     }
    #     nth_agg = prev_sorted.groupby('SK_ID_CURR').head(number).copy()
    #     nth_agg['IS_REFUSED'] = (nth_agg['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    #     nth_agg['IS_REVOLING'] = (nth_agg['NAME_CONTRACT_STATUS'] == 'Revolving loans').astype(int)
    #     nth_agg = nth_agg.groupby('SK_ID_CURR').agg(last_nth_agg)
    #     nth_agg.columns = pd.Index([f"last_{number}th_" +  e[0] + "_" + e[1].upper() for e in nth_agg.columns.tolist()])
    #     prev_agg = prev_agg.merge(right=nth_agg.reset_index(), how="left", on="SK_ID_CURR")

    prev_agg = prev_agg.set_index('SK_ID_CURR')
    return prev_agg
