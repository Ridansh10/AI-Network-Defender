from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_defender(X_train, X_test, y_train, y_test):
    print("Training the baseline AI defender model...")

    # Initialize the Random Forest algorithm out of the scikit-learn library
    # here n_estimators=100 means it uses 100 mini decision-trees together for accurate guessing
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model using the training data and answer keys
    model.fit(X_train, y_train)

    # Test the model's accuracy on the unseen test data
    predictions = model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, predictions)

    print(f"Baseline Defender Accuracy: {baseline_accuracy * 100:.2f}%")

    # Return the trained model so we can pass it to the hacker attack simulator
    return model