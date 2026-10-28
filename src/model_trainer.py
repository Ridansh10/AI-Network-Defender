import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from art.estimators.classification import PyTorchClassifier
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class NetworkDefender(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def train_defender(X_train, X_test, y_train, y_test):
    print("Training the baseline AI defender model...")

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train).astype(np.int64)
    y_test = np.asarray(y_test).astype(np.int64)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = NetworkDefender(input_dim, num_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(input_dim,),
        nb_classes=num_classes,
        device_type="cpu"
    )

    classifier.fit(X_train, y_train, batch_size=256, nb_epochs=5)

    predictions = np.argmax(classifier.predict(X_test), axis=1)
    baseline_accuracy = accuracy_score(y_test, predictions)

    print(f"Baseline Defender Accuracy: {baseline_accuracy * 100:.2f}%")

    return classifier