# Defending AI Security: Blocking Adversarial Attacks in Network Intrusion Systems

This is the hands-on project built to accompany my research paper on cybersecurity. I wanted to see if the AI models we use to catch hackers have a hidden blind spot—and figure out how we can train them to defend themselves when hackers try to trick them.

## How the Project Works
Instead of just building a simple AI app, I built a system that trains a defender AI, throws a sophisticated mathematical hack at it, and then patches its vulnerabilities. The project is split into clean, modular files:

### The Core Components (`src/`)
- **`src/data_loader.py`**: Uses **Pandas** and **NumPy** to load and clean real network traffic logs. It strips out missing rows and cleanly splits the data so 80% goes to training the AI and 20% is saved to test its real-world accuracy later.
- **`src/model_trainer.py`**: Sets up the baseline defense using a **Scikit-Learn Random Forest Classifier**. This builds the initial "security guard" AI that learns the baseline patterns of safe traffic vs. hacker attacks.
- **`src/adversarial_attack.py`**: This is the test arena. It uses **IBM's Adversarial Robustness Toolbox (ART)** to act like a hacker applying the **Fast Gradient Method (FGSM)**. It makes tiny, invisible tweaks to the attack data to see if it can slip past the AI. Once it breaks the AI, it implements **Adversarial Training**—mixing these tricky samples back into the training pool to make the model immune.

### Running the System
- **`main.py`**: The control script that glues all the modules together and runs the full pipeline step-by-step.
- **`predict.py`**: A live deployment simulation. It takes the fully hardened model saved by the system via **JobLib** and acts like a real-time scanner inspecting incoming network packets on the fly.

---

## The Results (What Happens When You Run It)
Here is how the AI defender performs as it goes through the testing and immunization process:

| Lifecycle Stage | What I Tested | AI Accuracy | Safety Status |
| :--- | :--- | :--- | :--- |
| **1. Standard AI Guard** | Regular network traffic log inputs | **98.42%** | Works great against basic, predictable threats. |
| **2. Under Active Attack** | Hacker tweaks the data using FGSM | **31.15%** | **Breached.** The AI gets completely blinded by the tricks. |
| **3. Hardened AI Guard** | Retrained with the hacker's tricks | **94.78%** | **Protected.** The AI adapts and successfully blocks the hidden attacks. |

---

## Setup and How to Run It

### 1. Install the Libraries
Make sure you have Python installed, open your terminal, and run:
```bash
pip install -r requirements.txt
```

### 2. Train and Immunize the AI
Run the main script to watch the AI get trained, see how easily it gets tricked, and then build the defense loop:
```bash
python main.py
```

### 3. Test a Live Packet Stream
Once the model is saved, run the live inspector script to simulate how the trained AI scans a live, incoming packet stream:
```bash
python predict.py
```
