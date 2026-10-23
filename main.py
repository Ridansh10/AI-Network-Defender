from src.data_loader import load_and_clean_data
from src.model_trainer import train_defender
from src.adversarial_attack import run_adversarial_arena
import os

def main():
    print("--- STARTING AI NETWORK DEFENDER PROTOCOL ---\n")
    # Definining the path to your dataset file
    dataset_path = "data/network_traffic.csv"

    # Check if the data file is in the folder
    if not os.path.exists(dataset_path):
        print("Error: Please place your network dataset CSV file at: '{dataset_path}'")
        print("Hint: You can use a sample dataset from Kaggle (like NSL-KDD or CICIDS2017) to run the code!")
        return

    # Phase 1: Loading and cleaning the data using pandas
    X_train, X_test, y_train, y_test = load_and_clean_data(dataset_path)
    print("\n--------------------------------------------------\n")

    # Phase 2: Training the AI baseline defender using scikit-learn
    baseline_model = train_defender(X_train, X_test, y_train, y_test)
    print("\n--------------------------------------------------\n")

    # Phase 3 and 4: Launch the hacker attack and deploy the defense formula
    final_defended_model = run_adversarial_arena(baseline_model, X_train, X_test, y_train, y_test)

    print("\n--- PROTOCOL COMPLETE: DEFENSE IMMUNIZATION SUCCESSFUL ---\n")

# Test block
if __name__ == "__main__":
    main()