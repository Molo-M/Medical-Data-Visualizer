import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
height_m = df['height']/100
sq_height = height_m ** 2
bmi = df['weight']/sq_height
mask = bmi < 25
bmi[mask] = 0
mask = bmi >= 25
bmi[mask] = 1
df['overweight'] = bmi.astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1,
# make the value 0. If the value is more than 1, make the value 1.

# cholesterol:
chol = df['cholesterol'].astype(float)
chol[chol == 1] = 0
chol[chol > 1] = 1
df['cholesterol'] = chol.astype(int)

# glucose:
gluc = df['gluc'].astype(float)
gluc[gluc == 1] = 0
gluc[gluc > 1] = 1
df['gluc'] = gluc.astype(int)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from
    # 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_long = df.melt(id_vars=['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cardio'],
                      ignore_index=False)
    df_long = df_long.reset_index()
    df_long.rename(columns={'index': 'total'}, inplace=True)
    df_cat = df_long

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    # You will have to rename one of the columns for the catplot to work correctly.
    df_long = df_long.groupby(['cardio', 'variable', 'value'], as_index=False).count()
    df_cat = df_long

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', y='total', data=df_cat, hue='value', kind='bar', col='cardio').fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data:

    # diastolic pressure
    ap_lo = df['ap_lo'] <= df['ap_hi']

    # height is less than the 2.5th percentile
    lo_height = df['height'] >= df['height'].quantile(0.025)

    # height is more than the 97.5th percentile
    hi_height = df['height'] <= df['height'].quantile(0.975)

    # weight is less than the 2.5th percentile
    lo_weight = df['weight'] >= df['weight'].quantile(0.025)

    # weight is more than the 97.5th percentile
    hi_weight = df['weight'] <= df['weight'].quantile(0.975)

    df_heat = df[lo_height & lo_weight & hi_height & hi_weight & ap_lo]

    # Calculate the correlation matrix
    corr = df_heat.corr(method='pearson')

    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(9, 9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, annot=True, mask=mask, fmt='.1f', square=True, linewidths=1, center=0.07, )

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
