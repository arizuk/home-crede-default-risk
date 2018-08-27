import pandas as pd


prefix = "kernel-131"
df1 = pd.read_csv('./blends/WEIGHT_AVERAGE_RANK2.csv')
df2 = pd.read_csv('./experiments/131-lgbm-kfold-0.792780.csv')

ratios = [0.5]
for i, ratio in enumerate(ratios):
    target = df1['TARGET'] * (1 - ratio) + df2['TARGET'] * ratio
    f = 'submits/blend-{}-{}.csv'.format(prefix, ratio)
    print(f)
    pd.DataFrame({
        'SK_ID_CURR': df1['SK_ID_CURR'],
        'TARGET': target
    }).to_csv(f, index=False)