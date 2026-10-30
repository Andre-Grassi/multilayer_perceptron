import io
import numpy as np
import pandas as pd
from PIL import Image

from ..core.perceptron import Perceptron, FullyConnectedMLP
from ..core.activations import SigmoidFunction

MNIST_TEST_DATASET_PATH = "datasets/mnist/test.parquet"
MNIST_TRAIN_DATASET_PATH = "datasets/mnist/train.parquet"

train_df = pd.read_parquet(MNIST_TRAIN_DATASET_PATH)
num_train_samples = train_df.shape[0]

# Store the bytes of each image into an array
train_bytes = np.array([item["bytes"] for item in train_df["image"]])

# Convert labels to one hot encoded representation
train_labels = train_df["label"].to_numpy()
train_labels = np.eye(10)[train_labels]

# Convert each image to grayscale pixels, convert to array and flatten
train_pixels = np.array(
    [
        np.array(Image.open(io.BytesIO(img_bytes)).convert("L")).flatten()
        for img_bytes in train_bytes
    ]
)

hidden_layer = [Perceptron(784, SigmoidFunction) for _ in range(16)]
output_layer = [Perceptron(16, SigmoidFunction) for _ in range(10)]
mlp = FullyConnectedMLP(784, [hidden_layer, output_layer])

a_arr = np.empty(shape=(num_train_samples, 10))
for i, train_sample in enumerate(train_pixels):
    a_arr[i] = mlp.a(train_sample)

print(mlp.cost(train_labels, a_arr))
