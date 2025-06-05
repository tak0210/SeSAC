import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 데이터 불러오기
df = pd.read_csv('C:/Users/Admin/Desktop/공모전/김가영/train_heat.csv')

# [1] tm 컬럼명 맞추기(혹시 'tm' 또는 'train_heat.tm' 형태일 수 있음)
if 'tm' not in df.columns:
    tm_col = [c for c in df.columns if 'tm' in c][0]
    df.rename(columns={tm_col: 'tm'}, inplace=True)

# [2] branch_id 및 heat_demand 컬럼명 일치화
if 'branch_id' not in df.columns:
    b_col = [c for c in df.columns if 'branch' in c][0]
    df.rename(columns={b_col: 'branch_id'}, inplace=True)
if 'heat_demand' not in df.columns:
    hd_col = [c for c in df.columns if 'heat_demand' in c][0]
    df.rename(columns={hd_col: 'heat_demand'}, inplace=True)

# [3] 기간 필터링(2021010101 ~ 2021123123)
df['tm'] = df['tm'].astype(str)
df = df[df['tm'].astype(int).between(2021010101, 2021123123)]

# [4] -99 → NaN
df.replace(-99, np.nan, inplace=True)

# [5] 결측치 선형보간(모든 결측 컬럼)
for col in df.columns[df.isnull().any()]:
    df[col] = df[col].interpolate(method='linear', limit_direction='both')

# ------------------------------------------------------------------------
# **파생변수 생성 구간**

# 날짜 파싱
df['year'] = df['tm'].str[:4].astype(int)
df['month'] = df['tm'].str[4:6].astype(int)
df['day'] = df['tm'].str[6:8].astype(int)
df['hour'] = df['tm'].str[8:10].astype(int)
df['date'] = pd.to_datetime(df['tm'].str[:8])
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = (df['weekday'] >= 5).astype(int)

# 기상 변수 변화량(차분) & 이동평균(24시간)
for var in ['ta', 'si', 'ta_chi', 'ws', 'wd', 'rn_day', 'rn_hr1', 'heat_demand']:
    if var in df.columns:
        df[f'{var}_diff'] = df[var].diff().fillna(0)
        df[f'{var}_ma24'] = df[var].rolling(window=24, min_periods=1).mean()

# 습도 변화량/이동평균
if 'si' in df.columns:
    df['si_diff'] = df['si'].diff().fillna(0)
    df['si_ma24'] = df['si'].rolling(24, min_periods=1).mean()

# ------------------------------------------------------------------------
# **모델 입력/타겟 컬럼 정의**
# 직접적으로 예측에 쓰지 않을 컬럼 리스트
exclude_cols = [
    'tm', 'branch_id', 'date', 'year', 'day',  # 고유/비식별/중복
    'heat_demand'  # 타겟
]

feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]

X = df[feature_cols].copy()
y = df['heat_demand']

# MinMax 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# ------------------------------------------------------------------------
# **LSTM 시퀀스 데이터 분할**
SEQ_LEN = 24

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X.iloc[i:(i + seq_length)].values)
        ys.append(y.iloc[i + seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled_df, y, SEQ_LEN)

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# ------------------------------------------------------------------------
# **LSTM 모델 설계/학습/평가**
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
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
print(f"LSTM 모델 RMSE(파생변수 포함): {rmse:.4f}")


# ---------출력 결과---------
# LSTM 모델 RMSE(파생변수 포함): 13.0315
