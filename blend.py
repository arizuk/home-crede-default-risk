import pandas as pd


df1 = pd.read_csv('./blends/WEIGHT_AVERAGE_RANK2.csv')
df2 = pd.read_csv('./experiments/090-lgbm-avg-0.795109.csv')

ratios = [0.5]
for i, ratio in enumerate(ratios):
    print(ratio)
    target = df1['TARGET'] * (1 - ratio) + df2['TARGET'] * ratio
    f = 'submits/blends{}-{}.csv'.format(i, ratio)
    print(f)
    pd.DataFrame({
        'SK_ID_CURR': df1['SK_ID_CURR'],
        'TARGET': target
    }).to_csv(f, index=False)