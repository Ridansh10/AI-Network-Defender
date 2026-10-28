from src.data_loader import load_and_clean_data
from src.model_trainer import train_defender
from src.adversarial_attack import run_adversarial_arena
import os
import torch


def main():
    print("--- STARTING AI NETWORK DEFENDER PROTOCOL ---\n")

    dataset_path = "data/network_traffic.csv"

    if not os.path.exists(dataset_path):
        print(f"Error: Please place your network dataset CSV file at: '{dataset_path}'")
        print("Hint: You can use a sample dataset from Kaggle (like NSL-KDD or CICIDS2017) to run the code!")
        return

    # Phase 1: Load and clean the data
    X_train, X_test, y_train, y_test = load_and_clean_data(dataset_path)
    print("\n--------------------------------------------------\n")

    # Phase 2: Train the baseline AI defender
    baseline_model = train_defender(X_train, X_test, y_train, y_test)
    print("\n--------------------------------------------------\n")

    # Phase 3 and 4: Run FGSM attack and adversarial training
    final_defended_model = run_adversarial_arena(
        baseline_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Save the hardened PyTorch model
    os.makedirs("models", exist_ok=True)

    model_path = "models/hardened_defender.pt"
    torch.save(baseline_model.model.state_dict(), model_path)

    print(f"\nHardened defender model saved to: {model_path}")
    print("\n--- PROTOCOL COMPLETE: DEFENSE IMMUNIZATION SUCCESSFUL ---\n")


if __name__ == "__main__":
    main()