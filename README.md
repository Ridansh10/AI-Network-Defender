# AI Network Defender

This project tests whether an AI model used to detect network intrusions can be fooled by small, intentional changes to network traffic data. It also tests whether the model can become more resistant to these attacks through adversarial training.

The project uses the CICIDS2017 dataset. A neural network is trained to classify network traffic as either benign or malicious. The Fast Gradient Method (FGSM) is then used to create slightly modified versions of the data that try to fool the model. Finally, the model is retrained using these adversarial examples to improve its robustness.

## How It Works

### 1. Data Loading

`src/data_loader.py` loads the CICIDS2017 dataset using Pandas and NumPy.

The code removes missing values and infinite values, converts the labels into a binary format, and keeps only numeric features. The data is split into 80% training data and 20% testing data.

The features are then scaled using Scikit-Learn’s `StandardScaler`. The scaler is fitted on the training data only and then applied to the test data.

### 2. Training the Defender

`src/model_trainer.py` trains a Multi-Layer Perceptron (MLP) using PyTorch.

The model receives 78 network-traffic features and predicts whether a traffic record is benign or malicious. The trained model is wrapped using IBM’s Adversarial Robustness Toolbox (ART), which allows gradient-based attacks such as FGSM.

A fixed random seed is used to make the training process more reproducible.

### 3. FGSM Attack

`src/adversarial_attack.py` applies the Fast Gradient Method using ART.

FGSM makes a small change to each input in the direction that increases the model’s loss. In this project, epsilon is set to 0.1. The attack is tested on a 5,000-sample subset of the test data.

### 4. Adversarial Training

After the attack, the project creates adversarial examples from a 5,000-sample subset of the training data.

These examples are added to the original training data, and the model is retrained. This helps the model learn to handle slightly modified inputs and improves its resistance to FGSM-style attacks.

## Results

| Stage | What Was Tested | Accuracy | Result |
| :--- | :--- | ---: | :--- |
| Standard AI defender | Clean test data | **99.97%** | Strong performance on normal traffic |
| Under FGSM attack | Modified test data | **45.32%** | Large drop in accuracy |
| Hardened AI defender | Model retrained with adversarial examples | **99.58%** | Accuracy recovered under attack |

These results show that a model with very high normal accuracy can still be vulnerable to adversarial attacks. They also show that adversarial training can significantly improve robustness.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- PyTorch
- IBM Adversarial Robustness Toolbox (ART)
- JobLib

## Project Structure

```text
AI-Network-Defender/
│
├── main.py                     # Runs the full training, attack, and defense pipeline
├── predict.py                  # Loads the saved model and simulates packet inspection
├── requirements.txt            # Python dependencies
│
└── src/
    ├── data_loader.py          # Loads, cleans, splits, and scales the dataset
    ├── model_trainer.py        # Trains the PyTorch MLP defender
    └── adversarial_attack.py   # Applies FGSM and adversarial training
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

Run the live packet inspection simulation:

```bash
python predict.py
```

## Main Takeaway

A high accuracy score does not mean an AI security system is safe. My model reached 99.97% accuracy on clean data, but FGSM reduced its accuracy to 45.32%. After adversarial training, accuracy under attack rose to 99.58%.

This project shows why AI security systems should be tested against deliberate manipulation, not only evaluated on normal data.