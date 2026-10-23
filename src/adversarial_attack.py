from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os

def run_adversarial_arena(model: RandomForestClassifier, X_train, X_test, y_train, y_test):
    print("Setting up the Hacker Arena...")

    # Wrapping the trained scikit-learn model inside the IBM ART environment
    classifier = SklearnClassifier(model=model)

    # Set up the FGSM attack using the correct class
    # here eps (epsilon) = 0.1 controls the maximum amount a hacker can tweak the columns
    hacker_attack = FastGradientMethod(estimator=classifier, eps=0.1)

    # The Hacker Attack: Generates tweaked/disguised test data
    print("Launching adversarial data-tweaking attack on the test dataset...")
    X_test_tweaked = hacker_attack.generate(x=X_test)

    # Testing how badly the baseline AI gets fooled by the disguised traffic
    tweaked_predictions = model.predict(X_test_tweaked)
    attack_accuracy = accuracy_score(y_test, tweaked_predictions)
    print(f"AI Defender Accuracy under Attack: {attack_accuracy * 100:.2f}%")

    # Building the Solution: Adversarial Training Defense
    print("Deploying Defense: Retraining AI with the hacker's tricks...")
    # Generating tweaked data from the training set to teach the AI what tricks look like
    X_train_tweaked = hacker_attack.generate(x=X_train)

    # Combining original clean data + hacker tweaked data together
    X_train_defended = np.vstack((X_train, X_train_tweaked))
    y_train_defended = np.hstack((y_train, y_train)) # Duplicate answer keys for both sets

    # Retraining the model on the combined data
    model.fit(X_train_defended, y_train_defended)

    # Re-test the defended model against the hacker's tricks
    final_predictions = model.predict(X_test_tweaked)
    defended_accuracy = accuracy_score(y_test, final_predictions)
    print(f"Final Defended AI Accuracy against Attack: {defended_accuracy * 100:.2f}%")

    # Saving the trained model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/defended_network_ai.pkl')
    print("Defender AI model saved successfully to 'models/defended_network_ai.pkl'")

    return model