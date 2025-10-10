# ==============================================================================
# 0. 환경 설정 및 라이브러리 임포트
# ==============================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# TensorFlow Metal 플러그인 활성화
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU 메모리 증가 허용
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 로그 레벨 설정

# ==============================================================================
# 1. 데이터셋 불러오기
# ==============================================================================
# 로컬 환경에서 데이터 파일 경로를 지정해주세요.
H5_DATA_PATH = '/Users/loonatrium/Documents/developmentprac/sehyun/metr-la.h5'

try:
    df_traffic = pd.read_hdf(H5_DATA_PATH)
    traffic_data_raw = df_traffic.values
    print("데이터셋이 성공적으로 로드되었습니다.")
except FileNotFoundError:
    print(f"오류: 지정된 경로에 파일이 없습니다 -> '{H5_DATA_PATH}'")
    print("데모 실행을 위해 임의의 데이터를 생성합니다.")
    traffic_data_raw = np.random.rand(34272, 207) * 100

print(f"원본 데이터 형태: {traffic_data_raw.shape}")

# ==============================================================================
# 2. 데이터 전처리
# ==============================================================================
# 결측치 처리 (선형 보간법)
df_traffic_filled = pd.DataFrame(traffic_data_raw).interpolate(method='linear', axis=0)
traffic_filled = df_traffic_filled.values

# 데이터 정규화 (MinMaxScaler)
scaler = MinMaxScaler()
traffic_normalized = scaler.fit_transform(traffic_filled)

# ==============================================================================
# 3. Autoencoder 모델 정의
# ==============================================================================
input_dim = traffic_normalized.shape[1]
encoding_dim = 64  # 잠재 공간의 차원

# 입력 레이어
input_layer = Input(shape=(input_dim,))

# 인코더
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# 디코더
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Autoencoder 모델 정의
autoencoder = Model(input_layer, decoded)

# 인코더 모델 정의
encoder = Model(input_layer, encoded)

# Autoencoder 컴파일
autoencoder.compile(optimizer='adam', loss='mse')

# ==============================================================================
# 4. 모델 학습
# ==============================================================================
X_train = traffic_normalized

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.2
)

# ==============================================================================
# 5. 학습 결과 시각화
# ==============================================================================
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ==============================================================================
# 6. 잠재 공간 표현 추출
# ==============================================================================
encoded_data = encoder.predict(X_train)
print(f"잠재 공간 데이터 형태: {encoded_data.shape}")
