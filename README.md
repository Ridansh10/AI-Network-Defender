# AI Network Defender

An offline experiment exploring how adversarial attacks affect a neural-network classifier trained to distinguish benign from malicious network traffic.

## Overview

This project uses a PyTorch multilayer perceptron (MLP) to classify network-traffic records as benign or malicious. It then evaluates the model’s robustness against Fast Gradient Sign Method (FGSM) adversarial attacks and examines whether adversarial fine-tuning improves resilience.

The project is an educational research experiment. It is **not** a live intrusion-detection or network-blocking system.

## Dataset

- **Dataset:** CICIDS2017
- **File used:** One Friday CSV file
- **Records processed:** 286,096
- **Features used:** 78 numeric features
- **Task:** Binary classification — benign vs. malicious traffic

## Approach

1. Load and preprocess one CICIDS2017 CSV file.
2. Select and prepare numeric features for classification.
3. Train a PyTorch MLP classifier.
4. Evaluate clean-data performance.
5. Generate FGSM adversarial examples.
6. Measure model accuracy under different attack strengths.
7. Test adversarial fine-tuning.
8. Run a fresh-attack epsilon sweep and visualize accuracy changes.

## Results

The model achieved strong classification performance on clean data. However, evaluation under newly generated adversarial examples showed that performance can decline as attack strength increases.

This highlights an important limitation: high accuracy on clean data, or on previously generated adversarial examples, does not necessarily prove that a model is robust against new adversarial attacks.

## Key Learning

- Adversarial robustness must be evaluated using freshly generated attacks, not only examples used during training or earlier testing.
- Clean accuracy alone is insufficient for assessing the reliability of machine-learning security systems.
- Adversarial training can improve resistance to certain attacks, but it does not make a model universally secure.

## Tools

- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

## How to Run

1. Clone the repository.
2. Install the required dependencies.
3. Place the required CICIDS2017 CSV file in the expected data directory.
4. Run the notebook or training script.
5. Review the generated accuracy and epsilon-sweep results.

## Limitations

- Uses one CICIDS2017 CSV file rather than the complete multi-day dataset.
- Performs offline classification and evaluation; it does not monitor or block live network traffic.
- Results depend on the selected dataset file, preprocessing choices, and model configuration.

## Future Improvements

- Expand evaluation across additional CICIDS2017 files.
- Compare additional adversarial attack methods.
- Improve preprocessing and class-balance handling.
- Document reproducible training and evaluation settings in more detail.