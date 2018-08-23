import pandas as pd


df1 = pd.read_csv('./blends/WEIGHT_AVERAGE_RANK2.csv')
df2 = pd.read_csv('./blends/084-lgbm-kfold-0.790403.csv')

ratios = [0.4, 0.6]
for i, ratio in enumerate(ratios):
    print(ratio)
    target = df1['TARGET'] * (1 - ratio) + df2['TARGET'] * ratio
    pd.DataFrame({
        'SK_ID_CURR': df1['SK_ID_CURR'],
        'TARGET': target
    }).to_csv('submits/blends{}-{}.csv'.format(i, ratio), index=False)