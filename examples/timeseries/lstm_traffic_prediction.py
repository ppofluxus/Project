"""
LSTM traffic prediction example.

Refactored from tests/Lstm_traffic_prediction.py:
- Hardcoded paths replaced with CLI args/env vars
- Safe font handling (fallback when Pretendard missing)
- Main guard and minimal logging
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Tuple


def set_env() -> None:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def set_font(plt_module, try_paths: list[str]) -> None:
    import matplotlib.font_manager as fm

    chosen = None
    for p in try_paths:
        if os.path.exists(p):
            try:
                name = fm.FontProperties(fname=p).get_name()
                plt_module.rc("font", family=name)
                chosen = p
                break
            except Exception:
                pass
    plt_module.rcParams["axes.unicode_minus"] = False
    if chosen:
        print(f"[font] Using Pretendard at {chosen}")
    else:
        print("[font] Pretendard not found, using default font")


def create_sequences(data: Any, input_steps: int, output_steps: int) -> Tuple[Any, Any]:
    import numpy as np

    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i : (i + input_steps)])
        y.append(data[(i + input_steps) : (i + input_steps + output_steps)])
    return np.array(X), np.array(y)


def build_model(params: dict, input_shape: Tuple[int, int], output_shape: Tuple[int, int]):
    from tensorflow.keras.layers import LSTM, Dense, Reshape, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=input_shape),
        LSTM(params["hidden_units"], activation="tanh"),
        Dense(output_shape[0] * output_shape[1], activation="relu"),
        Reshape(output_shape),
    ])
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss="mse")
    return model


def evaluate_metrics(y_true: Any, y_pred: Any, scaler: Any):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true_unscaled = scaler.inverse_transform(y_true.reshape(-1, y_true.shape[2]))
    y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, y_true.shape[2]))
    rmse = float(np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled)))
    mae = float(mean_absolute_error(y_true_unscaled, y_pred_unscaled))
    r2 = float(r2_score(y_true_unscaled, y_pred_unscaled))
    return rmse, mae, r2


def main():
    import matplotlib.pyplot as plt  # noqa: WPS433
    import numpy as np  # noqa: WPS433
    import pandas as pd  # noqa: WPS433
    from sklearn.preprocessing import MinMaxScaler  # noqa: WPS433

    parser = argparse.ArgumentParser(description="LSTM traffic prediction example")
    parser.add_argument("--h5", dest="h5_path", default=os.getenv("TRAFFIC_H5", ""), help="Path to metr-la.h5 (optional)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output-steps", type=int, default=6)
    parser.add_argument("--seq-grid", nargs="*", type=int, default=[6, 12, 24])
    parser.add_argument("--hidden-grid", nargs="*", type=int, default=[32, 64, 128])
    parser.add_argument("--batch-grid", nargs="*", type=int, default=[16, 32])
    parser.add_argument("--lr-grid", nargs="*", type=float, default=[1e-3, 1e-4])
    parser.add_argument("--results-dir", default=os.getenv("RESULTS_DIR", "examples/results"))
    parser.add_argument("--no-show", action="store_true", help="Do not show plots (save instead)")
    args = parser.parse_args()

    set_env()
    font_candidates = [
        os.getenv("PRETENDARD_PATH", ""),
        os.path.expanduser("~/Library/Fonts/Pretendard-Medium.otf"),
        os.path.expanduser("~/Library/Fonts/Pretendard-Regular.otf"),
    ]
    set_font(plt, [p for p in font_candidates if p])

    # Load data
    traffic_data_raw: np.ndarray
    if args.h5_path and os.path.exists(args.h5_path):
        df = pd.read_hdf(args.h5_path)
        traffic_data_raw = df.values
        print("[data] Loaded H5 data")
    else:
        print("[data] H5 not found; generating random demo data")
        traffic_data_raw = np.random.rand(34272, 207) * 100

    print(f"[data] shape={traffic_data_raw.shape}")

    # Preprocess
    df_filled = pd.DataFrame(traffic_data_raw).interpolate(method="linear", axis=0)
    traffic_filled = df_filled.values
    scaler = MinMaxScaler()
    traffic_normalized = scaler.fit_transform(traffic_filled)

    # Initial temp sequence to estimate sizes
    temp_input_steps = 12
    X_temp, y_temp = create_sequences(traffic_normalized, temp_input_steps, args.output_steps)
    train_size = int(len(X_temp) * 0.6)
    val_size = int(len(X_temp) * 0.2)

    param_grid = {
        "sequence_length": args.seq_grid,
        "hidden_units": args.hidden_grid,
        "batch_size": args.batch_grid,
        "learning_rate": args.lr_grid,
    }

    best_val_loss = float("inf")
    best_params: dict = {}
    print("[tune] start grid search")
    for seq_len in param_grid["sequence_length"]:
        X_tune, y_tune = create_sequences(traffic_normalized[:train_size], seq_len, args.output_steps)
        X_val, y_val = create_sequences(traffic_normalized[train_size : train_size + val_size], seq_len, args.output_steps)
        input_shape = (X_tune.shape[1], X_tune.shape[2])
        output_shape = (y_tune.shape[1], y_tune.shape[2])
        for hidden in param_grid["hidden_units"]:
            for batch in param_grid["batch_size"]:
                for lr in param_grid["learning_rate"]:
                    current = {"sequence_length": seq_len, "hidden_units": hidden, "batch_size": batch, "learning_rate": lr}
                    model = build_model(current, input_shape, output_shape)
                    history = model.fit(X_tune, y_tune, epochs=10, batch_size=batch, validation_data=(X_val, y_val), verbose=0)
                    val_loss = float(min(history.history["val_loss"]))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = current
                        print(f"[tune] best so far {best_params} val_loss={best_val_loss:.6f}")

    print(f"[tune] best={best_params} val_loss={best_val_loss:.6f}")

    # Final cumulative training
    final_input_steps = best_params["sequence_length"] if best_params else 12
    X_all, y_all = create_sequences(traffic_normalized, final_input_steps, args.output_steps)
    train_size = int(len(X_all) * 0.8)
    X_train_final, X_test_final = X_all[:train_size], X_all[train_size:]
    y_train_final, y_test_final = y_all[:train_size], y_all[train_size:]
    n_splits = 3
    split_indices = np.array_split(np.arange(len(X_train_final)), n_splits)

    results_lstm = []
    cumulative_data_indices: list[int] = []
    for i in range(n_splits):
        cumulative_data_indices.extend(split_indices[i])
        current_X = X_train_final[cumulative_data_indices]
        current_y = y_train_final[cumulative_data_indices]
        input_shape = (current_X.shape[1], current_X.shape[2])
        output_shape = (y_test_final.shape[1], y_test_final.shape[2])
        model_final = build_model(best_params or {"hidden_units": 64, "learning_rate": 1e-3, "batch_size": 32}, input_shape, output_shape)
        model_final.fit(current_X, current_y, epochs=args.epochs, batch_size=(best_params or {"batch_size": 32})["batch_size"], verbose=0)
        y_pred = model_final.predict(X_test_final, verbose=0)
        rmse, mae, r2 = evaluate_metrics(y_test_final, y_pred, scaler)
        results_lstm.append({"session": i + 1, "cumulative_data_size": len(current_X), "RMSE": rmse, "MAE": mae, "R2_Score": r2})
        print(f"[session {i+1}] RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    # Plot
    df_results = pd.DataFrame(results_lstm)
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df_results["session"], df_results["RMSE"], label="RMSE", marker="o")
    plt.plot(df_results["session"], df_results["MAE"], label="MAE", marker="o")
    plt.title("최적 모델의 누적 학습에 따른 성능 변화")
    plt.xlabel("학습 차시"); plt.ylabel("평가 지표"); plt.grid(True); plt.legend(); plt.tight_layout()
    fig_path = os.path.join(out_dir, "lstm_results.png")
    plt.savefig(fig_path, dpi=150)
    if not args.no_show:
        plt.show()
    csv_path = os.path.join(out_dir, "lstm_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[saved] {fig_path}, {csv_path}")


if __name__ == "__main__":
    main()
