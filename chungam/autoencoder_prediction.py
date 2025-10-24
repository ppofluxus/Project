# ==============================================================================
# 0. 환경 설정 및 라이브러리 임포트 (PyTables 없이 동작)
# ==============================================================================

# urllib3 LibreSSL 경고 억제 (macOS 기본 Python/LibreSSL 조합에서 자주 발생)
import warnings as _warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning as _NotOpenSSLWarning
    _warnings.simplefilter("ignore", _NotOpenSSLWarning)
except Exception:
    pass

import os
import math
import pickle
import json
from datetime import datetime
from typing import Optional, Dict, Any
import time
import itertools
import numpy as np
import pandas as pd
import h5py
from tqdm.auto import tqdm

import tensorflow as tf
from keras import Model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from keras.utils import disable_interactive_logging
from sklearn.preprocessing import MinMaxScaler

# Keras 기본 로그(내장 진행바) 비활성화 -> tqdm와 중복 방지
try:
    disable_interactive_logging()
except Exception:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU(MPS) 메모리 증가 허용 및 디바이스 안내
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"[INFO] GPU(MPS) 사용 가능: {len(gpus)}개 감지")
    else:
        print("[WARN] GPU(MPS) 디바이스가 감지되지 않았습니다. CPU로 실행됩니다.")
except Exception as e:
    print("[WARN] GPU 설정 중 예외:", e)


# ==============================================================================
# 1. 데이터셋 불러오기 (PyTables 불필요: h5py 전용 + pkl 로더)
# ==============================================================================

# 환경변수로 경로 지정 가능
H5_PATH = os.getenv("TRAFFIC_H5", "/Users/ppofluxus/Documents/Project/chungam/sehyun/METR-LA.h5")
PKL_PATH = os.getenv("TRAFFIC_PKL", "")
# 기본을 'batch'로 설정해 실시간 감을 높입니다. (필요 시 'epoch' 또는 'none')
PROGRESS_DETAIL = os.getenv("PROGRESS_DETAIL", "batch")  # 'batch' | 'epoch' | 'none'
GRID_LOG = os.getenv("GRID_LOG", "grid_progress.log")
GRID_CSV = os.getenv("GRID_CSV", "results_ae_grid.csv")
LIVE_STATUS = os.getenv("LIVE_STATUS", "live_status.txt")

# 시드 고정
np.random.seed(42)
tf.random.set_seed(42)


def _try_pickle_load(path: str):
    for enc in (None, "latin1", "bytes"):
        try:
            with open(path, "rb") as f:
                if enc is None:
                    return pickle.load(f)
                return pickle.load(f, encoding=enc)
        except Exception:
            continue
    with open(path, "rb") as f:
        return pickle.load(f)


def load_from_pkl(path: str) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"pkl 경로가 유효하지 않습니다: {path}")
    obj = _try_pickle_load(path)

    # ndarray
    if isinstance(obj, np.ndarray):
        return obj

    # dict 후보 키
    if isinstance(obj, dict):
        for k in ["data", "speed", "x", "values", "series"]:
            if k in obj:
                arr = obj[k]
                if isinstance(arr, list):
                    arr = np.array(arr)
                if isinstance(arr, np.ndarray):
                    return arr
        keys = ", ".join(map(str, obj.keys()))
        raise ValueError(
            "pkl에 시계열 배열 키를 찾지 못했습니다. 포함 키: " + keys
        )

    # (sensor_ids, id_map, adj_mx) 등은 여기서 처리하지 않음 (인접행렬만일 수 있음)
    raise ValueError(f"지원하지 않는 pkl 객체 타입: {type(obj)}")


def load_from_h5(path: str) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"H5 경로가 유효하지 않습니다: {path}")

    def _all_datasets(hf):
        for name in hf:
            obj = hf[name]
            if isinstance(obj, h5py.Dataset):
                yield name, obj
            elif isinstance(obj, h5py.Group):
                for subname, sub in _all_datasets(obj):
                    yield f"{name}/{subname}", sub

    with h5py.File(path, 'r') as hf:
        best = None
        best_score = -1
        for name, ds in _all_datasets(hf):
            shp = ds.shape
            if not shp:
                continue
            # 선호 점수: 2D(T,N) > 1D(T) > 3D(T,N,F)
            if len(shp) == 2 and shp[0] >= 50 and shp[1] >= 1:
                score = shp[0] * shp[1]
            elif len(shp) == 1 and shp[0] >= 50:
                score = shp[0]
            elif len(shp) >= 3 and shp[0] >= 50 and shp[1] >= 1:
                score = shp[0] * shp[1]
            else:
                score = -1
            if score > best_score:
                best_score = score
                best = (name, ds)
        if best is None:
            raise ValueError("h5py로 적절한 시계열 dataset을 찾지 못했습니다.")
        name, ds = best
        arr = ds[...]
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        elif arr.ndim == 2:
            return arr[:, 0:1]
        else:
            arr = arr[:, 0:1, ...]
            arr = arr.reshape(arr.shape[0], -1)
            return arr[:, 0:1]


def load_series() -> np.ndarray:
    # 우선 h5py 경로
    if H5_PATH and os.path.exists(H5_PATH):
        try:
            return load_from_h5(H5_PATH)
        except Exception as e:
            print("[WARN] h5py 로딩 실패, pkl 시도:", e)
    # pkl 경로 시도
    if PKL_PATH and os.path.exists(PKL_PATH):
        return load_from_pkl(PKL_PATH)
    raise FileNotFoundError("TRAFFIC_H5 또는 TRAFFIC_PKL 경로가 유효하지 않습니다.")


raw_arr = load_series()
print(f"원본(시계열) 데이터 형태: {raw_arr.shape}")

# 1채널 시계열로 정규화 (T, 1) 가정
if raw_arr.ndim == 1:
    series = raw_arr.reshape(-1, 1)
elif raw_arr.ndim == 2:
    series = raw_arr[:, 0:1]
elif raw_arr.ndim >= 3:
    series = raw_arr[:, 0:1, ...]
    series = series.reshape(series.shape[0], -1)[:, 0:1]
else:
    raise ValueError(f"예상치 못한 배열 차원: {raw_arr.shape}")

# 결측치 간단 보간
series_df = pd.DataFrame(series).interpolate(method='linear', axis=0)
series = series_df.values

# 정규화
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)


# ==============================================================================
# 2. 시퀀스 생성 및 데이터 분할 (시간 순 정렬 유지)
# ==============================================================================

TIMESTEPS = int(os.getenv("TIMESTEPS", "50"))

def create_sequences(data: np.ndarray, timesteps: int) -> np.ndarray:
    X = []
    for i in range(len(data) - timesteps):
        X.append(data[i:(i + timesteps)])
    return np.array(X)


sequences = create_sequences(series_scaled, TIMESTEPS)

# 3등분: train/val/test = 60/20/20
train_size = int(len(sequences) * 0.6)
val_size = int(len(sequences) * 0.2)

x_train_full = sequences[:train_size]
x_val = sequences[train_size:train_size + val_size]
x_test = sequences[train_size + val_size:]

print(f"전체 시퀀스 수: {len(sequences)}")
print(f"훈련: {len(x_train_full)}, 검증: {len(x_val)}, 테스트: {len(x_test)}")


# ==============================================================================
# 3. LSTM AutoEncoder 정의
# ==============================================================================

def create_lstm_autoencoder(timesteps: int, features: int, latent_dim: int, activation='tanh') -> Model:
    inputs = Input(shape=(timesteps, features))
    z = LSTM(latent_dim, activation=activation)(inputs)
    z = RepeatVector(timesteps)(z)
    y = LSTM(latent_dim, activation=activation, return_sequences=True)(z)
    outputs = TimeDistributed(Dense(features))(y)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


FEATURES = x_train_full.shape[-1]


# ==============================================================================
# 진행 상황 로깅 콜백 (tqdm + 파일 로그)
# ==============================================================================

class TQDMKerasCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs: int, steps_per_epoch: int, desc: str = "Training", mode: str = "epoch", log_path: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.desc = desc
        self.mode = mode  # 'batch' | 'epoch' | 'none'
        self.epoch_pbar = None
        self.batch_pbar = None
        self.log_path = log_path
        self.meta = meta or {}
        self._last_time = None
        self._avg_step_time = None  # EMA of step time

    def _log(self, event: str, payload: dict):
        if not self.log_path:
            return
        rec = {
            "time": datetime.utcnow().isoformat() + "Z",
            "event": event,
            **(self.meta or {}),
            **payload,
        }
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def on_train_begin(self, logs=None):
        if self.mode in ("batch", "epoch"):
            self.epoch_pbar = tqdm(total=self.epochs, desc=self.desc, leave=False)
        if self.mode == "batch":
            self.batch_pbar = tqdm(total=self.steps_per_epoch, desc=f"{self.desc}-batches", leave=False)
        self._log("train_begin", {"epochs": self.epochs, "steps_per_epoch": self.steps_per_epoch})
        self._last_time = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        now = time.perf_counter()
        dt = None
        if self._last_time is not None:
            dt = max(1e-6, now - self._last_time)
            if self._avg_step_time is None:
                self._avg_step_time = dt
            else:
                # EMA smoothing
                self._avg_step_time = 0.1 * dt + 0.9 * self._avg_step_time
        self._last_time = now

        # 추정 처리량 계산 (samples/sec, MB/s)
        bs = None
        try:
            if isinstance(self.meta.get("params"), dict):
                bs = int(self.meta["params"].get("batch_size"))
        except Exception:
            bs = None
        tsteps = int(self.meta.get("timesteps", 0))
        feats = int(self.meta.get("features", 0))
        samples_per_sec = (bs / self._avg_step_time) if (bs and self._avg_step_time) else None
        mbps = (samples_per_sec * tsteps * feats * 4 / (1024 * 1024)) if samples_per_sec else None

        if self.mode == "batch" and self.batch_pbar is not None:
            if loss is not None:
                postfix = {"loss": f"{loss:.6f}"}
                if samples_per_sec:
                    postfix.update({
                        "sps": f"{samples_per_sec:,.1f}",
                        "MB/s": f"{mbps:,.2f}" if mbps else "-",
                    })
                self.batch_pbar.set_postfix(postfix)
            self.batch_pbar.update(1)

        # ETA(에폭) 추정 및 라이브 상태 파일 갱신
        eta_epoch = None
        if self._avg_step_time:
            remain = max(0, self.steps_per_epoch - (batch + 1))
            eta_epoch = remain * self._avg_step_time
        status_line = {
            "phase": self.meta.get("phase"),
            "cfg": f"{self.meta.get('cfg_index')}/{self.meta.get('cfg_total')}" if self.meta.get("phase") == "grid" else None,
            "session": self.meta.get("session") if self.meta.get("phase") == "session" else None,
            "epoch_total": self.epochs,
            "epoch_progress": None,  # 채움은 epoch_end에서
            "batch": batch + 1,
            "steps_per_epoch": self.steps_per_epoch,
            "loss": float(loss) if loss is not None else None,
            "samples_per_sec": samples_per_sec,
            "mb_per_sec": mbps,
            "eta_epoch_sec": eta_epoch,
        }
        try:
            with open(LIVE_STATUS, "w") as f:
                f.write(json.dumps(status_line, ensure_ascii=False))
        except Exception:
            pass
        self._log("batch_end", {
            "batch": batch + 1,
            "loss": float(loss) if loss is not None else None,
            "samples_per_sec": samples_per_sec,
            "mb_per_sec": mbps,
            "eta_epoch_sec": eta_epoch,
        })

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if self.mode in ("batch", "epoch") and self.epoch_pbar is not None:
            if loss is not None:
                self.epoch_pbar.set_postfix(loss=f"{loss:.6f}")
            self.epoch_pbar.update(1)
        if self.mode == "batch" and self.batch_pbar is not None:
            self.batch_pbar.close()
            self.batch_pbar = tqdm(total=self.steps_per_epoch, desc=f"{self.desc}-batches", leave=False)
        # 라이브 상태 파일에 에폭 진행률 반영
        try:
            with open(LIVE_STATUS, "r") as f:
                cur = json.loads(f.read() or "{}")
        except Exception:
            cur = {}
        cur.update({"epoch_progress": int(epoch) + 1})
        try:
            with open(LIVE_STATUS, "w") as f:
                f.write(json.dumps(cur, ensure_ascii=False))
        except Exception:
            pass
        self._log("epoch_end", {"epoch": int(epoch) + 1, "loss": float(loss) if loss is not None else None})

    def on_train_end(self, logs=None):
        if self.batch_pbar is not None:
            self.batch_pbar.close()
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
        self._log("train_end", {})


# ==============================================================================
# 4. 그리드 서치 (비지도 복원오차 기반)
# ==============================================================================

param_grid = {
    'latent_dim': [256, 512, 1024],
    'batch_size': [16, 32, 64],
    'activation': ['relu', 'tanh', 'sigmoid'],
}

keys, values = zip(*param_grid.items())
experiment_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_cfg = None
best_val_rmse = float('inf')

print(f"\n총 {len(experiment_params)}개의 하이퍼파라미터 조합으로 그리드 서치를 시작합니다.")
for idx, params in enumerate(tqdm(experiment_params, desc="Grid", unit="cfg"), start=1):
    act_name = params['activation'].__name__ if callable(params['activation']) else str(params['activation'])
    tqdm.write(f"CFG {idx}: ld={params['latent_dim']} bs={params['batch_size']} act={act_name}")

    ae = create_lstm_autoencoder(TIMESTEPS, FEATURES, params['latent_dim'], params['activation'])
    epochs_grid = int(os.getenv('EPOCHS_GRID', '20'))
    steps_per_epoch = math.ceil(len(x_train_full) / params['batch_size'])
    cb = TQDMKerasCallback(
        epochs=epochs_grid,
        steps_per_epoch=steps_per_epoch,
        desc=f"Grid[{idx}/{len(experiment_params)}] ld={params['latent_dim']} bs={params['batch_size']} act={act_name}",
        mode=PROGRESS_DETAIL,
        log_path=GRID_LOG,
        meta={"phase": "grid", "cfg_index": idx, "cfg_total": len(experiment_params), "params": params, "timesteps": TIMESTEPS, "features": FEATURES},
    )
    ae.fit(
        x_train_full, x_train_full,
        epochs=epochs_grid,
        batch_size=params['batch_size'],
        shuffle=True, verbose=0,
        callbacks=[cb],
    )

    x_val_pred = ae.predict(x_val, verbose=0)
    val_mse = np.mean(np.power(x_val - x_val_pred, 2), axis=(1, 2))
    val_rmse = float(np.sqrt(np.mean(val_mse)))

    tqdm.write(f"  -> val_RMSE={val_rmse:.6f}")
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_cfg = params

    # 그리드 중간 결과 누적 저장
    try:
        row = params.copy()
        row.update({"val_RMSE": val_rmse})
        header_needed = not os.path.exists(GRID_CSV)
        pd.DataFrame([row]).to_csv(GRID_CSV, mode='a', header=header_needed, index=False)
    except Exception as _e:
        tqdm.write(f"[WARN] 그리드 결과 저장 실패: {_e}")

print("\n[그리드 최적 조합]")
print(best_cfg, "val_RMSE=", round(best_val_rmse, 6))


# ==============================================================================
# 5. 차시별(누적) 학습 및 테스트 RMSE 기록
# ==============================================================================

results_ae = []  # {session, train_size, rmse}
SESSIONS = int(os.getenv("SESSIONS", "5"))  # 5차시: 20/40/60/80/100% 누적

for s in range(1, SESSIONS + 1):
    frac = s / SESSIONS
    n_train = max(1, int(len(x_train_full) * frac))
    x_train_cum = x_train_full[:n_train]

    tqdm.write(f"\n[차시 {s}/{SESSIONS}] 누적 학습 데이터: {n_train}")
    ae = create_lstm_autoencoder(TIMESTEPS, FEATURES, best_cfg['latent_dim'], best_cfg['activation'])
    epochs_final = int(os.getenv('EPOCHS_FINAL', '50'))
    steps_per_epoch = math.ceil(len(x_train_cum) / best_cfg['batch_size'])
    cb = TQDMKerasCallback(
        epochs=epochs_final,
        steps_per_epoch=steps_per_epoch,
        desc=f"Session[{s}/{SESSIONS}] train={n_train}",
        mode=PROGRESS_DETAIL,
        log_path=GRID_LOG,
        meta={"phase": "session", "session": s, "sessions": SESSIONS, "params": best_cfg, "timesteps": TIMESTEPS, "features": FEATURES},
    )
    ae.fit(
        x_train_cum, x_train_cum,
        epochs=epochs_final,
        batch_size=best_cfg['batch_size'],
        shuffle=True, verbose=0,
        callbacks=[cb],
    )

    x_test_pred = ae.predict(x_test, verbose=0)
    test_mse = np.mean(np.power(x_test - x_test_pred, 2), axis=(1, 2))
    test_rmse = float(np.sqrt(np.mean(test_mse)))

    results_ae.append({
        'session': s,
        'train_size': n_train,
        'latent_dim': best_cfg['latent_dim'],
        'batch_size': best_cfg['batch_size'],
        'activation': best_cfg['activation'] if isinstance(best_cfg['activation'], str) else best_cfg['activation'].__name__,
        'RMSE': test_rmse,
    })

    # 중간 저장 (안전)
    try:
        pd.DataFrame(results_ae).to_csv('results_ae_sessions.csv', index=False)
    except Exception as _e:
        tqdm.write(f"[WARN] 결과 저장 실패: {_e}")

print("\n[차시별 RMSE 요약]")
print(pd.DataFrame(results_ae).to_string(index=False))

