# ==============================================================================
# 0. 환경 설정 및 라이브러리 임포트
# ==============================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# TensorFlow Metal 플러그인 비활성화
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU 메모리 증가 허용
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to use CPU only
# os.environ['TF_METAL_DISABLE'] = '1'  # Disable TensorFlow Metal plugin
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 로그 레벨 설정

# Pretendard 폰트 경로 설정
font_path_candidates = [
    "/Users/loonatrium/Library/Fonts/Pretendard-Medium.otf"  # macOS 사용자 및 시스템 폰트 경로
]

font_path = None
for path in font_path_candidates:
    if os.path.exists(path):
        font_path = path
        break

if not font_path:
    raise FileNotFoundError("Pretendard 폰트를 찾을 수 없습니다. 폰트를 설치하거나 경로를 확인하세요.")

pretendard_font = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=pretendard_font)

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
# 2. 시계열 전처리
# ==============================================================================
# 결측치 처리 (선형 보간법)
df_traffic_filled = pd.DataFrame(traffic_data_raw).interpolate(method='linear', axis=0)
traffic_filled = df_traffic_filled.values

# 데이터 정규화 (MinMaxScaler)
scaler = MinMaxScaler()
traffic_normalized = scaler.fit_transform(traffic_filled)

# 시퀀스 생성 함수 (Sliding Window)
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:(i + input_steps)])
        y.append(data[(i + input_steps):(i + input_steps + output_steps)])
    return np.array(X), np.array(y)

# 데이터 분할을 위한 기본 시퀀스 길이 설정
temp_input_steps = 12
output_steps = 6

X_temp, y_temp = create_sequences(traffic_normalized, temp_input_steps, output_steps)

# 학습용:검증용:평가용 데이터 분리 (6:2:2 비율)
train_size = int(len(X_temp) * 0.6)
val_size = int(len(X_temp) * 0.2)

X_train_full, X_val_full, X_test_full = X_temp[:train_size], X_temp[train_size:train_size+val_size], X_temp[train_size+val_size:]
y_train_full, y_val_full, y_test_full = y_temp[:train_size], y_temp[train_size:train_size+val_size], y_temp[train_size+val_size:]

print(f"\n전체 학습 데이터셋 크기: {len(X_train_full)}")
print(f"검증 데이터셋 크기: {len(X_val_full)}")
print(f"평가 데이터셋 크기: {len(X_test_full)}")

# ==============================================================================
# 3. 하이퍼파라미터 튜닝 (Grid Search)
# ==============================================================================
print("\n" + "="*50)
print("3. 지도학습 LSTM 모델 하이퍼파라미터 튜닝")
print("="*50)

param_grid = {
    'sequence_length': [6, 12, 24],
    'hidden_units': [32, 64, 128],
    'batch_size': [16, 32],
    'learning_rate': [1e-3, 1e-4]
}

def build_model(params, input_shape, output_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(params['hidden_units'], activation='tanh'),
        Dense(output_shape[0] * output_shape[1], activation='relu'),
        Reshape(output_shape)
    ])
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse')
    return model

best_val_loss = float('inf')
best_params = {}

print("하이퍼파라미터 튜닝을 시작합니다...")
for seq_len in param_grid['sequence_length']:
    X_tune, y_tune = create_sequences(traffic_normalized[:train_size], seq_len, output_steps)
    X_val, y_val = create_sequences(traffic_normalized[train_size:train_size+val_size], seq_len, output_steps)

    input_shape = (X_tune.shape[1], X_tune.shape[2])
    output_shape = (y_tune.shape[1], y_tune.shape[2])

    for hidden in param_grid['hidden_units']:
        for batch in param_grid['batch_size']:
            for lr in param_grid['learning_rate']:
                current_params = {
                    'sequence_length': seq_len,
                    'hidden_units': hidden,
                    'batch_size': batch,
                    'learning_rate': lr
                }
                print(f"\n테스트 중: {current_params}")

                model = build_model(current_params, input_shape, output_shape)

                history = model.fit(X_tune, y_tune,
                                    epochs=10,
                                    batch_size=current_params['batch_size'],
                                    validation_data=(X_val, y_val),
                                    verbose=0)

                val_loss = min(history.history['val_loss'])
                print(f"-> 검증 손실(val_loss): {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = current_params
                    print(f"*** 새로운 최적 파라미터 발견! -> {best_params} (손실: {best_val_loss:.6f}) ***")

print("\n--- 하이퍼파라미터 튜닝 종료 ---")
print(f"최종 최적 파라미터: {best_params}")
print(f"최종 최적 검증 손실: {best_val_loss:.6f}")

# ==============================================================================
# 4. 최종 모델 누적 학습 및 평가
# ==============================================================================
print("\n" + "="*50)
print("4. 최적 파라미터로 최종 모델 누적 학습 및 평가")
print("="*50)

final_input_steps = best_params['sequence_length']
X_all, y_all = create_sequences(traffic_normalized, final_input_steps, output_steps)

train_size = int(len(X_all) * 0.8)
X_train_final, X_test_final = X_all[:train_size], X_all[train_size:]
y_train_final, y_test_final = y_all[:train_size], y_all[train_size:]

n_splits = 3
split_indices = np.array_split(np.arange(len(X_train_final)), n_splits)

def evaluate_metrics(y_true, y_pred, scaler):
    y_true_unscaled = scaler.inverse_transform(y_true.reshape(-1, y_true.shape[2]))
    y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, y_true.shape[2]))
    rmse = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    return rmse, mae, r2

results_lstm = []
cumulative_data_indices = []

for i in range(n_splits):
    print(f"\n--- 차시 {i+1} ---")

    cumulative_data_indices.extend(split_indices[i])
    current_X_train = X_train_final[cumulative_data_indices]
    current_y_train = y_train_final[cumulative_data_indices]

    print(f"사용된 누적 학습 데이터 크기: {len(current_X_train)}")

    final_input_shape = (current_X_train.shape[1], current_X_train.shape[2])
    final_output_shape = (y_test_final.shape[1], y_test_final.shape[2])
    model_final = build_model(best_params, final_input_shape, final_output_shape)

    model_final.fit(current_X_train, current_y_train,
                    epochs=50,
                    batch_size=best_params['batch_size'],
                    verbose=0)

    y_pred_lstm = model_final.predict(X_test_final)
    rmse, mae, r2 = evaluate_metrics(y_test_final, y_pred_lstm, scaler)

    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2 Score: {r2:.4f}")

    results_lstm.append({
        'session': i + 1,
        'cumulative_data_size': len(current_X_train),
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2
    })

# ==============================================================================
# 5. 결과 시각화
# ==============================================================================
df_results_lstm = pd.DataFrame(results_lstm)

print("\n" + "="*50)
print("5. 결과 시각화")
print("="*50)

plt.figure(figsize=(12, 6))
plt.plot(df_results_lstm['session'], df_results_lstm['RMSE'], label='RMSE', marker='o')
plt.plot(df_results_lstm['session'], df_results_lstm['MAE'], label='MAE', marker='o')
plt.title('최적 모델의 누적 학습에 따른 성능 변화', fontsize=16)
plt.xlabel('학습 차시', fontsize=12)
plt.ylabel('평가 지표', fontsize=12)
plt.xticks(df_results_lstm['session'])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()