import pandas as pd
import numpy as np
from src.utils import logit

@logit
def encode_categories(train, test, y, features):
    for f in features:
        mean_target_encoding(train, test, y, f)
        count_encoding(train, test, y, f)
        # train[f], indexer = pd.factorize(train[f])
        # test[f] = indexer.get_indexer(test[f])

def mean_target_encoding(train, test, y, column):
    df = pd.DataFrame({})
    df['TARGET'] = y
    df[column] = train[column]

    agg = df.groupby(column).TARGET.agg(['mean'])
    train[column] = train[column].map(agg['mean'])
    test[column] = test[column].map(agg['mean'])

def count_encoding(train, test, y, column):
    df = pd.DataFrame({})
    df['TARGET'] = y
    df[column] = train[column]

    counts = df.groupby(column).TARGET.count()
    train["X_{}_COUNT".format(column)] = train[column].map(counts)
    test["X_{}_COUNT".format(column)] = test[column].map(counts)

def income_median(train, test, column):
    medians = train.groupby(column).AMT_INCOME_TOTAL.median()
    col_median = "X_{}_INCOME_MEDIAN".format(column)
    train[col_median] = train[column].map(medians)
    test[col_median] = test[column].map(medians)

    col_ratio = "X_{}_INCOME_MEDIAN_RATIO".format(column)
    train[col_ratio] = np.log(train['AMT_INCOME_TOTAL'] / train[col_median])
    test[col_ratio] = np.log(test['AMT_INCOME_TOTAL'] / test[col_median])

    # means = train.groupby(column).X_EXT_SOURCES_MEAN.mean()
    # col_mean = "X_{}_EXT_SOURCES_MEAN".format(column)
    # train[col_mean] = train[column].map(means)
    # test[col_mean] = test[column].map(means)

    # col_ratio = "X_{}_EXT_SOURCES_RATIO".format(column)
    # train[col_ratio] = train[col_mean] / train['X_EXT_SOURCES_MEAN']
    # test[col_ratio] = test[col_mean] / test['X_EXT_SOURCES_MEAN']

# def combine_categories(df):
#     list_to_combine = [
#         ['OCCUPATION_TYPE', 'NAME_INCOME_TYPE'],
#         ['OCCUPATION_TYPE', 'CODE_GENDER'],
#         ['OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE'],
#         ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'],
#     ]
#     for categs in list_to_combine:
#         new_col = '_'.join(categs)
#         df[new_col] = 'X'
#         for categ in categs:
#             df[new_col] = df[new_col] + "_" + df[categ]

# def merge_minor_category(train, test, col):
#     minor = train[col].value_counts() < 3000
#     # keep NaN asis
#     train.loc[train[col].map(minor).fillna(False), col] = '-'
#     test.loc[test[col].map(minor).fillna(False), col] = '-'

def app_features(df):
    df['X_AMT_LOAN_PERIOD'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['X_AMT_GOODS_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    # df['AMT_GOODS_DIFF'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    # df['AMT_CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['X_AMT_ANNUITY_INCOME_RATIO_LOG'] =  np.log((df['AMT_INCOME_TOTAL']/df['AMT_ANNUITY']).fillna(0)+0.001)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    # Social features
    df['X_WORKING_LIFE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['X_INCOME_PER_FAM'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    df['X_HOUR_APPR_PROCESS_START'] = df.HOUR_APPR_PROCESS_START.astype('category')
    del df['HOUR_APPR_PROCESS_START']

    df['X_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['X_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['X_SCORES_STD'] = df['X_SCORES_STD'].fillna(df['X_SCORES_STD'].mean())

    doc_flags = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    for f in doc_flags:
        del df[f]

    # df['X_OCCUPATION_TYPE'] = df.OCCUPATION_TYPE.astype('category')
    # del df['OCCUPATION_TYPE']

    columns = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
      'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',
      'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
      'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
      'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
      'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',
      'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
      ]

    for c in columns:
      del df[c]

def prev_features(df):
    # AMT APPLICATION
    diff = (df['AMT_APPLICATION'] - df['AMT_CREDIT']) / df['AMT_CREDIT']
    df['X_AMT_APPLICATION_DIFF_RATIO'] = diff
    del df['AMT_APPLICATION']

def combined_features(df):
    df['X_APPROVTED_AMT_CREDIT_RATIO'] = df['AMT_CREDIT'] / df['prev_X_MAX_AMT_CREDIT']
    return True