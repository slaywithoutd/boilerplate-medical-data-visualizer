import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')
print(df.head())

# 2
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)
print(df.head())

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5:
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],  
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']  
    )

    # 6
    df_cat = df_cat.value_counts(['cardio', 'variable', 'value']).reset_index(name='total')

    # 7
    g = sns.catplot(
    x='variable', y='total', hue='value', col='cardio',
    data=df_cat, kind='bar', height=5, aspect=1,
    order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )

    # 8
    fig = g.figure

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Diastolic pressure <= Systolic pressure
        (df['height'] >= df['height'].quantile(0.025)) &  # Height >= 2.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &  # Height <= 97.5th percentile
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Weight >= 2.5th percentile
        (df['weight'] <= df['weight'].quantile(0.975))    # Weight <= 97.5th percentile
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        square=True,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )


    # 16
    fig.savefig('heatmap.png')
    return fig
