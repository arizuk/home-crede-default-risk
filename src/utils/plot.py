import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def percentage_plot(data, x):
    y = 'percent'
    hue = 'TARGET'

    prop_df = (data[x]
            .groupby(data[hue])
            .value_counts(normalize=True)
            .rename(y)
            .reset_index())

    plt.figure(figsize = (10,5))
    sns.barplot(x=x, y=y, hue=hue, data=prop_df)
    # labelは回転できる
    # if(label_rotation):
    #     s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()

def distplot(data, x):
    plt.figure(figsize = (10,5))
    sns.distplot(data[data.TARGET == 0][x].dropna(), kde_kws={ "label": 'target=0' })
    sns.distplot(data[data.TARGET == 1][x].dropna(),kde_kws={ "label": 'target=1' })
    plt.show()

def sort_values(values):
    if type(values) == pd.core.arrays.categorical.Categorical:
        return values.sort_values()
    else:
        return np.sort(values)

def percentage_plot2(data, x, order_by='value_count'):
    fig, axs = plt.subplots(ncols=2, figsize=(20, 5))

    df = percentage(data, x)

    order = {}
    if order_by == 'categorical':
        order['order'] = sort_values(df[x].values)
    if order_by == 'value_count':
        order['order'] = df[x].values

    s = sns.barplot(x=x, y='percent', data=df, ax=axs[0], **order)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=10)

    df = target_percentage(data, x)
    s = sns.barplot(x=x, y='percent', data=df, ax=axs[1], **order)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.show()

def percentage_plot3(data, x):
    df1 = percentage(data, x)
    df1.columns = [x, 'Percent']
    df2 = target_percentage(data, x)
    df2.columns = [x, 'TargetPercent']
    df1 = df1.merge(right=df2, on=x).sort_values(by='Percent', ascending=False)
    df1.set_index(x).plot.bar(figsize=(20, 8))
    plt.show()

def percentage(data, x):
    df = data[x].value_counts(normalize=True)
    df = df.reset_index()
    df.columns = [x, 'percent']
    return df

def target_percentage(data, x):
    hue = 'TARGET'
    df = data[[x, hue]].groupby([x], as_index=False).mean()
    df = pd.DataFrame({
        x: df[x],
        'percent': df[hue],
    })
    return df.sort_values(by=['percent'], ascending=False).reset_index(drop=True)