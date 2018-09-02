import pandas as pd


name = "corr-151-129"
# df1 = pd.read_csv('./blends/WEIGHT_AVERAGE_RANK2.csv')
df1 = pd.read_csv('./blends/corr_blend.csv')
df2 = pd.read_csv('./experiments/151-lgbm-kfold-0.793128.csv')
df3 = pd.read_csv('./experiments/129-lgbm-kfold-0.793110.csv')
# df3 = pd.read_csv('./experiments/132-lgbm-avg-0.795612.csv')

# ratios = [0.5]
# for i, ratio in enumerate(ratios):

target = df1['TARGET'] * 0.3 + df2['TARGET'] * 0. + df3['TARGET'] * 0.3
f = 'submits/blend-{}.csv'.format(name)
print(f)
pd.DataFrame({
    'SK_ID_CURR': df1['SK_ID_CURR'],
    'TARGET': target
}).to_csv(f, index=False)