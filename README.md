# Hypoglycemia Detection from ECG Signals

A deep learning web app that detects hypoglycemic episodes from raw ECG signals. Upload a CSV file and get an instant prediction.

![App Screenshot](Screenshot_2026-04-01_at_11_15_33_AM.png)

---

## About the Project

This project was developed as part of the **Deep Learning course at the Lebanese American University (LAU)**. The model was built as a team effort, and the deployment (Flask app + AWS EC2) was done independently.

The idea is simple: given a 10-second ECG signal, can we detect whether the patient is experiencing a hypoglycemic event? Hypoglycemia is dangerous and often goes undetected, so building an automated detection system from ECG data has real clinical value.

---

## How It Works

1. The user uploads a CSV file containing a raw ECG signal (2500 data points)
2. The signal gets normalized and fed into the model
3. The model outputs a probability score
4. If the score is above 0.70, the patient is flagged as **Hypoglycemic**

---

## The Model

We trained and compared two architectures:
- **ResNet** (CNN with residual blocks) — 120K parameters
- **CNN-LSTM** (hybrid) — 46K parameters

The ResNet model was selected for deployment due to its superior sensitivity (83.33%), which is the most critical metric in a clinical setting — missing a hypoglycemic event is far more dangerous than a false alarm.

**Training approach:** The dataset was heavily imbalanced (5% positive cases), so we augmented the positive class using noise injection, time shifting, amplitude scaling, and time warping — growing the positive samples from 1,723 to 24,174.

**Final model performance (test set):**
| Metric | Score |
|--------|-------|
| AUROC | 0.9666 |
| Sensitivity | 83.33% |
| Specificity | 94.68% |
| F1-Score | 0.6025 |

---

## Tech Stack

- **Model:** TensorFlow / Keras
- **Back-end:** Flask (Python)
- **Front-end:** HTML, CSS, JavaScript
- **Deployment:** AWS EC2 (Ubuntu 22.04)

---

## Project Structure

```
├── Notebook_Group2.ipynb        # Model training and evaluation
├── app.py                       # Flask back-end
├── templates/
│   └── index.html               # Front-end UI
└── README.md
```

---

## Running the App Locally

**1. Clone the repo:**
```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

**2. Install dependencies:**
```bash
pip install flask tensorflow numpy pandas scipy
```

**3. Add your model file** (`hypoglycemia_resnet_augmented_70_10_20.keras`) to the root folder

**4. Run the app:**
```bash
python app.py
```

**5. Open your browser at:** `http://localhost:5000`

---

## Input Format

The input CSV file should contain 2500 columns named `0` to `2499`, where each value represents a data point in the ECG signal sampled over 10 seconds.
