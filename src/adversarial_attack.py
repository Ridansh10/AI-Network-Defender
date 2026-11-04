import numpy as np
from art.attacks.evasion import FastGradientMethod
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os


def run_adversarial_arena(model, X_train, X_test, y_train, y_test):
    print("Setting up the Hacker Arena...")

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train).astype(np.int64)
    y_test = np.asarray(y_test).astype(np.int64)

    # Use samples so FGSM generation stays fast
    X_train_sample = X_train[:5000]
    X_test_sample = X_test[:5000]
    y_train_sample = y_train[:5000]
    y_test_sample = y_test[:5000]

    baseline_accuracy = accuracy_score(
        y_test_sample,
        np.argmax(model.predict(X_test_sample), axis=1)
    )
    print(f"Original model accuracy on clean data: {baseline_accuracy * 100:.2f}%")

    print("Launching FGSM adversarial attack...")
    hacker_attack = FastGradientMethod(estimator=model, eps=0.1)

    print("Generating adversarial examples from the test set...")
    X_test_tweaked = hacker_attack.generate(x=X_test_sample)

    print("Testing the original defender against adversarial examples...")
    tweaked_predictions = np.argmax(model.predict(X_test_tweaked), axis=1)
    attack_accuracy = accuracy_score(y_test_sample, tweaked_predictions)

    print(f"Accuracy under FGSM attack: {attack_accuracy * 100:.2f}%")

    print("Training the defender with adversarial examples...")
    X_train_tweaked = hacker_attack.generate(x=X_train_sample)

    X_train_defended = np.vstack((X_train_sample, X_train_tweaked))
    y_train_defended = np.hstack((y_train_sample, y_train_sample))

    model.fit(X_train_defended, y_train_defended, batch_size=256, nb_epochs=3)

    print("Testing the defended model...")
    final_predictions = np.argmax(model.predict(X_test_tweaked), axis=1)
    defended_accuracy = accuracy_score(y_test_sample, final_predictions)

    print(f"Defended model accuracy under FGSM attack: {defended_accuracy * 100:.2f}%")

    print("Classification report under FGSM attack:")
    print(classification_report(y_test_sample, tweaked_predictions))

    print("Confusion matrix under FGSM attack:")
    print(confusion_matrix(y_test_sample, tweaked_predictions))

    return {
        "baseline_accuracy": baseline_accuracy,
        "attack_accuracy": attack_accuracy,
        "defended_accuracy": defended_accuracy,
        "X_test_tweaked": X_test_tweaked
    }

def run_epsilon_sweep(model, X_test, y_test):
    print("Running FGSM epsilon sweep...")

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test).astype(np.int64)

    X_test_sample = X_test[:5000]
    y_test_sample = y_test[:5000]

    eps_values = [0.05, 0.1, 0.2, 0.3]
    accuracies = []

    for eps in eps_values:
        print(f"Testing FGSM with epsilon = {eps}...")

        attack = FastGradientMethod(estimator=model, eps=eps)
        X_attacked = attack.generate(x=X_test_sample)

        predictions = np.argmax(model.predict(X_attacked), axis=1)
        accuracy = accuracy_score(y_test_sample, predictions)

        accuracies.append(accuracy)

        print(f"Accuracy at epsilon {eps}: {accuracy * 100:.2f}%")

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8, 15))
    plt.plot(eps_values, accuracies, marker="o")
    plt.title("MLP accuracy under FGSM attacks")
    plt.xlabel("FGSM Epsilon")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("results/fgsm_epsilon_sweep.png")
    plt.close()

    print("Chart saved to results/fgsm_epsilon_sweep.png")

    return eps_values, accuracies