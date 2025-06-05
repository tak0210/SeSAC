import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense      # GRU만 추가/변경
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 데이터 불러오기
df = pd.read_csv('C:/Users/Admin/Desktop/공모전/김가영/train_heat.csv')

# 'train_heat.tm' 컬럼 값이 2021010101 부터 20211231까지인 데이터만 추출
df['train_heat.tm'] = df['train_heat.tm'].astype(str)
df = df[df['train_heat.tm'].astype(int).between(2021010101, 2021123123)]  # 시간 정보까지 반영

# -99를 np.nan으로 변환
df.replace(-99, np.nan, inplace=True)

# 결측치가 있는 컬럼 자동 탐색
missing_cols = df.columns[df.isnull().any()].tolist()
print("결측치가 있는 컬럼:", missing_cols)
print(df[missing_cols].isnull().sum())

# 결측치가 있는 컬럼에만 선형보간(양방향)
for col in missing_cols:
    df[col] = df[col].interpolate(method='linear', limit_direction='both')

print(df[missing_cols].isnull().sum())

#########################################################################################

# 파생변수 예시 (여기선 모든 숫자형 변수만 사용)
feature_cols = [col for col in df.columns if col not in ['train_heat.tm', 'train_heat.branch_id', 'train_heat.heat_demand']]

X = df[feature_cols]
y = df['train_heat.heat_demand']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

print("입력 특성 shape:", X_scaled_df.shape)
print("타겟 shape:", y.shape)

#########################################################################################

SEQ_LEN = 24

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X.iloc[i:(i + seq_length)].values)
        ys.append(y.iloc[i + seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled_df, y, SEQ_LEN)

print("시퀀스 입력 shape:", X_seq.shape)
print("시퀀스 타겟 shape:", y_seq.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#########################################################################################

# ★ LSTM → GRU로만 변경!
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
print(f"GRU 모델 RMSE: {rmse:.4f}")
 

#  GRU 모델 RMSE: 113.3535