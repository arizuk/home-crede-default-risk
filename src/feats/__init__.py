import pandas as pd


def simplify_hour_appr_process_start(v):
  if v >= 22 or v == 0:
    return 1
  elif v >= 1 and v <= 9:
    return 2
  else:
    return 3

def append_features(app_df):
  app_df['AMT_LOAN_PERIOD'] = app_df['AMT_CREDIT'] / app_df['AMT_ANNUITY']
  app_df['AMT_GOODS_RATIO'] = app_df['AMT_GOODS_PRICE'] / app_df['AMT_CREDIT']
  app_df['C_HOUR_APPR_PROCESS_START'] = app_df.HOUR_APPR_PROCESS_START.apply(simplify_hour_appr_process_start)