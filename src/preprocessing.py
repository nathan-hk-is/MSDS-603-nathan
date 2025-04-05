# Nathan Holmes-King
# 2025-04-04

import numpy as np
import pandas as pd

df = pd.read_csv('data/reference-data.csv')
df = df[['animal-life-stage', 'animal-sex', 'animal-comments']]
df['animal-life-stage'] = df['animal-life-stage'].str.split(' ').str[0]
df['animal-sex'] = df['animal-sex'].apply(lambda x: 1 if x == 'm' else 0
                                          if x == 'f' else np.NaN)
df['prey'] = df['animal-comments'].str.split('prey_p_month: ').str[1]
df = df.dropna()
X = df[['animal-life-stage', 'animal-sex']]
y = df['prey']

X.to_csv('data/X.csv')
y.to_csv('data/y.csv')
