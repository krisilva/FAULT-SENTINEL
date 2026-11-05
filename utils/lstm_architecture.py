"""
Fault Sentinel — LSTM Architecture
Rebuilds the exact LSTM architecture used during training (window=10,
9 features, 10 output classes) so we can load *weights only*
(lstm_prediction_model.weights.h5) instead of a full .keras file.

Why: loading a full .keras file with tf.keras.models.load_model() bakes
in the exact Keras version's serialization format. If the machine that
trained the model has a different TensorFlow/Keras version than the
machine running the dashboard, deserialization can fail (e.g. newer
Keras adding config keys like 'quantization_config' that an older Keras
does not recognise, or vice versa). Rebuilding the architecture in code
and loading only the numeric weights sidesteps that entirely, so the
app works across TensorFlow/Keras versions.
"""

WINDOW = 10
N_FEATURES = 9
N_CLASSES = 10


def build_lstm_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential([
        Input(shape=(WINDOW, N_FEATURES)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(N_CLASSES, activation="softmax"),
    ])
    return model
