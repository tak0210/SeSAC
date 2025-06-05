from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("train_heat.csv", on_bad_lines='skip', encoding='utf-8', engine='python')
df = df[df['train_heat.tm'].astype(str).str.len() == 10]
df['datetime'] = pd.to_datetime(df['train_heat.tm'].astype(str), format='%Y%m%d%H')
df['train_heat.ta_chi'] = df['train_heat.ta_chi'].replace(-99.0, 0)
df['train_heat.heat_demand'] = df['train_heat.heat_demand'].fillna(0)

features = ['train_heat.ta_chi', 'train_heat.heat_demand']
X = df[features].dropna()
X_scaled = StandardScaler().fit_transform(X)

model = KMeans(n_clusters=4, random_state=42)
X['cluster'] = model.fit_predict(X_scaled)
print(X.head())
