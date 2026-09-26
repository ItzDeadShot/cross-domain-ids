"""Base model architectures.

Layer sizes and names match the models in ``models/`` exactly. Only the learning
rate (and the batch size, at fit time) were tuned.
"""
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, Reshape
from tensorflow.keras.models import Model

from .config import N_FEATURES

LEARNING_RATES = [0.0001, 0.001, 0.01]
BATCH_SIZES = [32, 64, 128, 256]


def build_mlp():
    inputs = Input(shape=(N_FEATURES,), name="MLP_input")
    x = Dense(24, activation="relu", name="mlp_dense_1")(inputs)
    x = Dense(16, activation="relu", name="mlp_dense_2")(x)
    outputs = Dense(1, activation="sigmoid", name="mlp_output")(x)
    return Model(inputs=inputs, outputs=outputs, name="Custom_MLP_Model")


def build_dnn():
    inputs = Input(shape=(N_FEATURES,), name="DNN_input_layer")
    x = Dense(32, activation="relu", name="DNN_dense_layer_1")(inputs)
    x = Dense(16, activation="relu", name="DNN_dense_layer_2")(x)
    x = Dense(8, activation="relu", name="DNN_dense_layer_3")(x)
    x = Dense(4, activation="relu", name="DNN_dense_layer_4")(x)
    x = Dense(2, activation="relu", name="DNN_dense_layer_5")(x)
    outputs = Dense(1, activation="sigmoid", name="DNN_output_layer")(x)
    return Model(inputs=inputs, outputs=outputs, name="DNN_Custom_Model")


def build_gru():
    inputs = Input(shape=(N_FEATURES,), name="GRU_reshape_input")
    x = Reshape(target_shape=(1, N_FEATURES), name="gru_reshape_layer")(inputs)
    x = GRU(24, activation="relu", return_sequences=True, name="gru_layer_1")(x)
    x = GRU(16, activation="relu", return_sequences=False, name="gru_layer_2")(x)
    x = Dense(8, activation="relu", name="gru_dense")(x)
    outputs = Dense(1, activation="sigmoid", name="gru_output_layer")(x)
    return Model(inputs=inputs, outputs=outputs, name="Custom_GRU_Model")


def build_lstm():
    inputs = Input(shape=(N_FEATURES,), name="LSTM_reshape_input")
    x = Reshape(target_shape=(1, N_FEATURES), name="lstm_reshape_layer")(inputs)
    x = LSTM(24, activation="relu", return_sequences=True, name="lstm_layer_1")(x)
    x = LSTM(16, activation="relu", return_sequences=False, name="lstm_layer_2")(x)
    x = Dense(8, activation="relu", name="lstm_dense")(x)
    outputs = Dense(1, activation="sigmoid", name="lstm_output_layer")(x)
    return Model(inputs=inputs, outputs=outputs, name="Custom_LSTM_Model")


BUILDERS = {"MLP": build_mlp, "DNN": build_dnn, "GRU": build_gru, "LSTM": build_lstm}


def build_model(arch: str, learning_rate: float) -> Model:
    model = BUILDERS[arch]()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
