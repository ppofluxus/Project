"""Lightweight smoke tests to ensure example modules import and main entry exists.
These tests do NOT run training to keep CI fast.
"""

import importlib


def test_import_timeseries_autoencoder():
    m = importlib.import_module("examples.timeseries.autoencoder_prediction")
    assert hasattr(m, "main")


def test_import_timeseries_lstm():
    m = importlib.import_module("examples.timeseries.lstm_traffic_prediction")
    assert hasattr(m, "main")


def test_import_vision_catvsdog():
    m = importlib.import_module("examples.vision.catvsdog")
    assert hasattr(m, "main")
