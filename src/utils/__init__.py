import os
import re
import pandas as pd
import pickle
import gc
import numpy as np
import json
import contextlib
from timeit import default_timer as timer
import humanfriendly
import functools
from src.utils import slack

EXPERIMENT_DIR = os.path.join('./experiments')
CACHE_DIR = os.path.join('./cache')

def save_result(config, test, test_preds, clf, features, kfold):
    auc = config['auc']
    model = config['model']

    exp_id = experiment_id()
    output_base = f"{exp_id}-{model}-{'kfold' if kfold else 'avg'}-{auc:.6f}"

    csv = os.path.join(EXPERIMENT_DIR, output_base + ".csv")
    config_json = os.path.join(EXPERIMENT_DIR, f"{exp_id}.json")
    feature_csv = os.path.join(EXPERIMENT_DIR, f"{exp_id}-features.csv")

    test['TARGET'] = test_preds
    test[['SK_ID_CURR', 'TARGET']].to_csv(csv, index=False, float_format='%.8f')
    print(f'Save to {csv}')

    slack.notify(output_base)

    with open(config_json, 'w') as fp:
        json.dump(config, fp, indent=2)

    if clf:
        importance_df = pd.DataFrame({ 'features': features })
        importance_df['importance'] = [clf.feature_importances_[features.index(f)] for f in features]
        importance_df = importance_df.sort_values(by=['importance'], ascending=False)
        importance_df.to_csv(feature_csv, index=False)


def experiment_id():
    files = [f for f in os.listdir(EXPERIMENT_DIR)]
    files.sort()
    files.reverse()
    for f in files:
        match = re.match('(\d+)(-.*)?.csv', f)
        if match:
            last_id = int(match.group(1)) + 1
            return f"{last_id:03d}"

    return '001'

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def read_csv(file):
    cache_file = os.path.join(CACHE_DIR, file + ".pkl")
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, 'rb'))

    cache_dir = os.path.dirname(cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(file)
    reduce_mem_usage(df)
    df.to_pickle(cache_file)
    gc.collect()
    return df


@contextlib.contextmanager
def timeit():
    start = timer()
    yield
    end = timer()
    print('Wall time: {}'.format(humanfriendly.format_timespan(end - start)), flush=True)

def logit(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        start = timer()
        result = func(*args,**kwargs)
        end = timer()
        elapsed = humanfriendly.format_timespan(end - start)
        print('{} done. {}'.format(func.__name__, elapsed), flush=True)
        return result
    return wrapper