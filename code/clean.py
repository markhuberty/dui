import pandas as pd

df = pd.read_csv('../data/alcdata.csv')
df = df[['sublastname', 'subfirstname', 'submi']]
df = df.dropna()
df.to_csv('../data/dedupe_alc_input.csv', index=False)
