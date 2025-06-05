import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/Admin/Desktop/공모전/김가영/train_heat.csv')
df['train_heat.tm'] = df['train_heat.tm'].astype(str)
df = df[df['train_heat.tm'].astype(int).between(2021010101, 2021123123)]
df.replace(-99, np.nan, inplace=True)
for col in df.columns[df.isnull().any()]:
    df[col] = df[col].interpolate(method='linear', limit_direction='both')

# ========== [기존 파생변수] ==========
df['month'] = df['train_heat.tm'].str[4:6].astype(int)
df['hour'] = df['train_heat.tm'].str[8:10].astype(int)
df['date'] = pd.to_datetime(df['train_heat.tm'].str[:8])
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = (df['weekday'] >= 5).astype(int)
for var in ['ta', 'si', 'ta_chi', 'ws', 'wd', 'rn_day', 'rn_hr1', 'train_heat.heat_demand']:
    if var in df.columns:
        df[f'{var}_diff'] = df[var].diff().fillna(0)
        df[f'{var}_ma24'] = df[var].rolling(window=24, min_periods=1).mean()

# ========== [추천 파생변수 추가] ==========

# 누적 강수량(최근 3/6/12/24시간)
for h in [3, 6, 12, 24]:
    if 'rn_hr1' in df.columns:
        df[f'rn_hr1_sum{h}'] = df['rn_hr1'].rolling(h, min_periods=1).sum()

# 최고/최저 기온(최근 6/12/24시간)
for h in [6, 12, 24]:
    if 'ta' in df.columns:
        df[f'ta_max{h}'] = df['ta'].rolling(h, min_periods=1).max()
        df[f'ta_min{h}'] = df['ta'].rolling(h, min_periods=1).min()

# 체감온도-실제온도 차
if 'ta_chi' in df.columns and 'ta' in df.columns:
    df['diff_ta_chi'] = df['ta_chi'] - df['ta']

# 풍속 급변(2-step 차분)
if 'ws' in df.columns:
    df['ws_diff2'] = df['ws'].diff(2).fillna(0)

# 당일 누적 강수량
if 'rn_day' in df.columns:
    df['rn_day_cumsum'] = df['rn_day'].cumsum()

# 전일 평균/최고/최저 기온
if 'ta' in df.columns:
    df['ta_yesterday_avg'] = df['ta'].shift(24).rolling(24).mean()
    df['ta_yesterday_max'] = df['ta'].shift(24).rolling(24).max()
    df['ta_yesterday_min'] = df['ta'].shift(24).rolling(24).min()

# 임계치 이탈 플래그
if 'ta' in df.columns:
    df['cold_flag'] = (df['ta'] < 5).astype(int)
    df['hot_flag'] = (df['ta'] > 25).astype(int)

# ========== [입력·타겟 컬럼] ==========
feature_cols = [
    col for col in df.columns
    if col not in ['train_heat.tm', 'train_heat.branch_id', 'train_heat.heat_demand', 'date']
    and df[col].dtype in [np.float64, np.int64]
]
X = df[feature_cols]
y = df['train_heat.heat_demand']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

SEQ_LEN = 24
def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X.iloc[i:(i + seq_length)].values)
        ys.append(y.iloc[i + seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled_df, y, SEQ_LEN)
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

model = Sequential([
    GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_test, y_test)
)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"GRU+파생변수 확장 모델 RMSE: {rmse:.4f}")

# ----------출력결과----------
# GRU+파생변수 확장 모델 RMSE: 9.0562