import os
import numpy as np
import torch

from src.data_loader import load_and_clean_data
from src.model_trainer import NetworkDefender


MODEL_PATH = "models/hardened_defender.pt"
DATASET_PATH = "data/network_traffic.csv"


def live_packet_inspector():
    print("\nInspecting incoming network packet stream...")

    if not os.path.exists(MODEL_PATH):
        print("Error: Saved model not found. Please run 'python main.py' first to train and save the model.")
        return

    # Load the dataset to get the correct input size and simulate live packets
    X_train, X_test, y_train, y_test = load_and_clean_data(DATASET_PATH)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # Load the hardened PyTorch defender
    model = NetworkDefender(input_dim, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print("Hardened AI defender loaded successfully.")

    # Simulate a small stream of incoming network packets
    incoming_packets = X_test[:10]

    with torch.no_grad():
        packet_tensor = torch.tensor(incoming_packets, dtype=torch.float32)
        outputs = model(packet_tensor)
        predictions = torch.argmax(outputs, axis=1).numpy()

    print("\nSecurity scan results:")

    for i, prediction in enumerate(predictions, start=1):
        if prediction == 0:
            print(f"[STATUS: SAFE] Packet {i}: behavior matches normal traffic parameters.")
        else:
            print(f"[STATUS: BLOCKED] Packet {i}: malicious anomaly detected by Immunized Defender AI.")

    print("\n--- LIVE PACKET INSPECTION COMPLETE ---")


if __name__ == "__main__":
    live_packet_inspector()