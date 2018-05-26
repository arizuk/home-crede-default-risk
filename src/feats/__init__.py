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

def sum_reg_region_and_city_flags(df):
    columns = [
      'REG_REGION_NOT_LIVE_REGION',
      'REG_REGION_NOT_WORK_REGION',
      'LIVE_REGION_NOT_WORK_REGION',
      'REG_CITY_NOT_LIVE_CITY',
      'REG_CITY_NOT_WORK_CITY',
      'LIVE_CITY_NOT_WORK_CITY',
    ]
    cnt = np.zeros(df.shape[0])
    for col in columns:
        cnt = df[col] + cnt

    for col in columns:
      del df[col]

    return cnt

def age_category(df):
  age = (df.DAYS_BIRTH * -1)/365
  ageC = pd.cut(age, list(range(0, 101, 5)), right=False)
  return ageC

def append_features(app_df):
    app_df['AMT_LOAN_PERIOD'] = app_df['AMT_CREDIT'] / app_df['AMT_ANNUITY']
    app_df['AMT_GOODS_RATIO'] = app_df['AMT_GOODS_PRICE'] / app_df['AMT_CREDIT']
    app_df['C_HOUR_APPR_PROCESS_START'] = app_df.HOUR_APPR_PROCESS_START.apply(
        simplify_hour_appr_process_start)
    del app_df['HOUR_APPR_PROCESS_START']
    app_df['C_AGE'] = age_category(app_df)

    # app_df['DOCS_FLAG'] = sum_document_flags(app_df).apply(lambda x: 1 if x >= 1 else 0)
    for i in range(2, 22):
        c = f'FLAG_DOCUMENT_{i}'
        del app_df[c]

    # app_df['REG_REGION_AND_CITY_DIFF_CNT'] = sum_reg_region_and_city_flags(app_df)