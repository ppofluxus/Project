"""
Autoencoder example for traffic data reconstruction.

Refactored from tests/autoencoder_prediction.py:
- Hardcoded paths replaced by CLI args / env var
- Main guard added; plotting optional
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def set_env() -> None:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def build_autoencoder(input_dim: int, encoding_dim: int = 64):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation="relu")(input_layer)
    encoded = Dense(encoding_dim, activation="relu")(encoded)
    decoded = Dense(128, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def main():
    parser = argparse.ArgumentParser(description="Autoencoder traffic example")
    parser.add_argument("--h5", dest="h5_path", default=os.getenv("TRAFFIC_H5", ""))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    set_env()

    # Load data
    if args.h5_path and os.path.exists(args.h5_path):
        df = pd.read_hdf(args.h5_path)
        traffic = df.values
        print("[data] Loaded H5")
    else:
        print("[data] H5 not found; generating random demo data")
        traffic = np.random.rand(34272, 207) * 100
    print(f"[data] shape={traffic.shape}")

    # Preprocess
    df_filled = pd.DataFrame(traffic).interpolate(method="linear", axis=0)
    traffic_filled = df_filled.values
    scaler = MinMaxScaler()
    traffic_norm = scaler.fit_transform(traffic_filled)

    input_dim = traffic_norm.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim)
    history = autoencoder.fit(
        traffic_norm,
        traffic_norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_split=args.val_split,
        verbose=1,
    )

    # Plot
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()
    if args.no_show:
        os.makedirs("examples/results", exist_ok=True)
        out = "examples/results/autoencoder_loss.png"
        plt.savefig(out, dpi=150)
        print(f"[saved] {out}")
    else:
        plt.show()

    encoded_data = encoder.predict(traffic_norm, verbose=0)
    print(f"[encoded] shape={encoded_data.shape}")


if __name__ == "__main__":
    main()
