import requests
import pandas as pd

df = pd.read_csv('../data/reference-data.csv').dropna()
df['animal-life-stage'] = df['animal-life-stage'].str.split(' ').str[0]
df['animal-sex'] = df['animal-sex'].apply(lambda x: 1 if x == 'm' else 0
                                          if x == 'f' else np.NaN)
df = df.dropna()
df = df[['animal-life-stage', 'animal-sex']]
comment = {'rows': df.to_dict(orient='records')}

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=comment)
print(response.json())
