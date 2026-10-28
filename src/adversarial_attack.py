import numpy as np
from sklearn.metrics import accuracy_score
from art.attacks.evasion import FastGradientMethod


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

    return {
        "baseline_accuracy": accuracy_score(
            y_test_sample,
            np.argmax(model.predict(X_test_sample), axis=1)
        ),
        "attack_accuracy": attack_accuracy,
        "defended_accuracy": defended_accuracy,
        "X_test_tweaked": X_test_tweaked
    }