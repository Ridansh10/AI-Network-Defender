import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def live_packet_inspector(new_packet_data):
    print("\n Inspecting incoming network packet stream...")

    # Loading the saved Defender AI model
    try:
        model: RandomForestClassifier = joblib.load('models/defended_network_ai.pkl')
    except FileNotFoundError:
        print("Error: Saved model not found. Please run 'python main.py' first to train and save the model.")
        return

    # Reshape the input data to match what the AI expects (1 row, multiple features) 
    packet_features = np.array(new_packet_data).reshape(1, -1)

    # Using the trained AI model to make a real-world prediction
    prediction = model.predict(packet_features)
    
    # Output the security result
    if prediction[0] == 0:
        print("[STATUS: SAFE] Packet behavior matches normal traffic parameters.")
    else:
        print("[STATUS: BLOCKED] Malicious anomaly detected! Packet isolated by Immunized Defender AI.")

# Test Block
# Simulating a live packet passing through
if __name__ == "__main__":
    # Let's say the dataset has 5 features
    sample_incoming_packet = [0.02, 1420, 0, 4, 12.5]
    live_packet_inspector(sample_incoming_packet)