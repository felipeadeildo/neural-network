import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from neural_network import DenseLayer, NeuralNetwork
from sklearn.preprocessing import StandardScaler

data_path = Path("data")
with zipfile.ZipFile(data_path / "digit-recognizer.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# Carregando e processando os dados
train = pd.read_csv(data_path / "train.csv")
X_train = train.iloc[:, 1:].values / 255.0  # Normalização simples
Y_train = train.iloc[:, 0].values.reshape(-1, 1)  # type: ignore

# One-hot encoding das labels
Y_train = np.eye(10)[
    Y_train.flatten()
].T  # One-hot encode e transpose para (n_classes, n_samples)

# Normalização dos dados de entrada
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).T  # Transpor para (n_features, n_samples)

# Definir as camadas da rede neural
layers = [
    DenseLayer(input_size=784, output_size=128, activation="relu"),
    DenseLayer(input_size=128, output_size=64, activation="relu"),
    DenseLayer(input_size=64, output_size=10, activation="softmax"),
]

# Inicializar a rede neural
nn = NeuralNetwork(layers=layers, learning_rate=0.013)

# Treinar a rede neural
nn.train(X_train, Y_train, epochs=25, verbose_freq=1)
