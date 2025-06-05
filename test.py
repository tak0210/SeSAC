from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_squared_error
import pandas as pd
import os

# 파일 나가는 경로 확인 ㄱㄱ
csv_file = "train_heat.csv"
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found in the current directory.")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    exit(1)

try:
    df = pd.read_csv("train_heat.csv", on_bad_lines='skip', encoding='utf-8', engine='python')
    print(f"Successfully loaded CSV with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Check if required columns exist
    required_columns = ['train_heat.tm', 'train_heat.ta_chi', 'train_heat.heat_demand']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
    
    df = df[df['train_heat.tm'].astype(str).str.len() == 10]
    print(f"After filtering by tm length: {len(df)} rows")
    
    df['datetime'] = pd.to_datetime(df['train_heat.tm'].astype(str), format='%Y%m%d%H')
    df['train_heat.ta_chi'] = df['train_heat.ta_chi'].replace(-99.0, 0)
    df['train_heat.heat_demand'] = df['train_heat.heat_demand'].fillna(0)
    
    features = ['train_heat.ta_chi', 'train_heat.heat_demand']
    X = df[features].dropna()
    print(f"After dropna: {len(X)} rows")
    print(f"Sample data:\n{X.head()}")
    
    if len(X) == 0:
        print("No data available for clustering after preprocessing.")
        exit(1)
    
    X_scaled = StandardScaler().fit_transform(X)
    
    model = KMeans(n_clusters=4, random_state=42)
    clusters = model.fit_predict(X_scaled)
    X_with_clusters = X.copy()
    X_with_clusters['cluster'] = clusters
    print(f"\nClustering results:\n{X_with_clusters.head()}")
    print(f"\nCluster distribution:\n{pd.Series(clusters).value_counts().sort_index()}")
    
    # Silhouette Score 계산
    sil_score = silhouette_score(X_scaled, clusters)
    print(f"\nSilhouette Score: {sil_score:.3f}")
    
    # 클러스터 중심값을 기준으로 RMSE 계산 (마지막 피처가 열수요여야 함)
    rmse = mean_squared_error(X[features[-1]], model.cluster_centers_[clusters][:, -1], squared=False)
    print(f"RMSE: {rmse:.3f}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 