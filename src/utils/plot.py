import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

def percentage_plot2(data, x):
    fig, axs = plt.subplots(ncols=2, figsize=(20, 5))

    df = percentage(data, x)
    order = df[x].values
    s = sns.barplot(x=x, y='percent', data=df, ax=axs[0], order=order)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=10)

    df = target_percentage(data, x)
    s = sns.barplot(x=x, y='percent', data=df, ax=axs[1], order=order)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.show()

def percentage(data, x):
    df = data[x].value_counts(normalize=True)
    df = df.reset_index()
    df.columns = [x, 'percent']
    return df

def target_percentage(data, x):
    hue = 'TARGET'
    df = data[[x, hue]].groupby([x], as_index=False).mean()
    return pd.DataFrame({
        'percent': df[hue],
        x: df[x],
    })