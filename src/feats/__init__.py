import pandas as pd
import numpy as np


def simplify_hour_appr_process_start(v):
    if v >= 22 or v == 0:
        return 1
    elif v >= 1 and v <= 9:
        return 2
    else:
        return 3


def sum_document_flags(df):
    cnt = np.zeros(df.shape[0])
    for i in range(2, 22):
        col = f'FLAG_DOCUMENT_{i}'
        cnt = df[col] + cnt
    return cnt

def age_category(df):
  age = (df.DAYS_BIRTH * -1)/365
  ageC = pd.cut(age, list(range(0, 101, 5)), right=False)
  return ageC

def app_features(df):
    df['AMT_LOAN_PERIOD'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['AMT_GOODS_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    # df['AMT_GOODS_DIFF'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    # df['AMT_CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    # df['AMT_ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['C_HOUR_APPR_PROCESS_START'] = df.HOUR_APPR_PROCESS_START.apply(
        simplify_hour_appr_process_start)
    del df['HOUR_APPR_PROCESS_START']

    for i in range(2, 22):
        c = f'FLAG_DOCUMENT_{i}'
        del df[c]

def prev_features(df):
    df['AMT_LOAN_PERIOD'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['AMT_GOODS_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['C_HOUR_APPR_PROCESS_START'] = df.HOUR_APPR_PROCESS_START.apply(
        simplify_hour_appr_process_start)
    del df['HOUR_APPR_PROCESS_START']