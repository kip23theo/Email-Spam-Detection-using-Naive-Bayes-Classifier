# 🛡️ AI Email Spam Detector

A machine learning web application that detects spam emails in real-time using Naive Bayes classifier, TF-IDF, and NLP preprocessing — deployed with Flask.

---

## 📊 Results

| Model | Accuracy | F1 Score |
|---|---|---|
| **Naive Bayes** | **98%** | **0.959** ✓ Best |
| Logistic Regression | 98% | 0.922 |
| SVM | 99% | 0.939 |
| Random Forest | 97% | 0.931 |

---

## 🚀 Features

- ✅ Real dataset — 5,574 SMS messages (SMS Spam Collection)
- ✅ NLP preprocessing — stemming, stopwords, URL/number replacement
- ✅ TF-IDF with bigrams for feature extraction
- ✅ 4 ML models compared with 5-fold cross-validation
- ✅ Live web UI with real-time predictions
- ✅ Confidence score with animated progress bar
- ✅ Spam trigger word highlighting
- ✅ Black & blue professional UI theme

---

## 🗂️ Project Structure

```
spam_detector/
├── data/
│   └── spam.csv            ← Dataset (SMS Spam Collection)
├── src/
│   ├── preprocess.py       ← NLP preprocessing pipeline
│   └── evaluate.py         ← Visualization dashboard
├── templates/
│   └── index.html          ← Web UI (HTML/CSS/JS)
├── model/
│   ├── spam_model.pkl      ← Saved best model
│   └── vectorizer.pkl      ← Saved TF-IDF vectorizer
├── venv/                   ← Virtual environment
├── app.py                  ← Flask web server
├── train_model.py          ← Model training script
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## ⚙️ Setup & Installation

### 1. Clone or download the project
```bash
cd Desktop/spam_detector
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate virtual environment
```bash
# Windows
venv\Scripts\activate.bat

# Mac/Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install numpy pandas scikit-learn nltk flask matplotlib seaborn wordcloud
```

---

## ▶️ How to Run

### Every time you open VS Code:

**Step 1** — Activate venv:
```bash
venv\Scripts\activate.bat
```

**Step 2** — Train the model (only needed once):
```bash
python train_model.py
```

**Step 3** — Start the web server:
```bash
python app.py
```

**Step 4** — Open browser and go to:
```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

```
Raw Email Text
      ↓
NLP Preprocessing
(lowercase → remove URLs → remove numbers → stem words → remove stopwords)
      ↓
TF-IDF Vectorization
(convert text to numerical features, 10,000 features, bigrams)
      ↓
Naive Bayes Classifier
(calculate P(spam|words) using Bayes theorem)
      ↓
Prediction + Confidence Score
(SPAM 🚫 or HAM ✅ with probability %)
```

---

## 📦 Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Pandas | Data loading & manipulation |
| NLTK | Stopwords, stemming (PorterStemmer) |
| Scikit-learn | ML models, TF-IDF, metrics |
| Flask | Web server & REST API |
| NumPy | Numerical computations |
| HTML/CSS/JS | Frontend web interface |
| Pickle | Model saving & loading |

---

## 📁 Dataset

- **Name:** SMS Spam Collection
- **Source:** Kaggle / UCI ML Repository
- **Size:** 5,574 messages
- **Spam:** 747 messages (13.4%)
- **Ham:** 4,827 messages (86.6%)
- **Format:** CSV with columns v1 (label) and v2 (message)

---

## 🔐 Cybersecurity Relevance

Spam emails are the #1 delivery mechanism for:
- Phishing attacks
- Malware and ransomware
- Social engineering

This system acts as a **first line of defense** — similar to filters used by Gmail, Outlook, and enterprise email gateways — automatically blocking malicious content before it reaches users.

---

## 📸 Web UI Features

- Paste any email or SMS text
- Click **Analyze Email** or press **Ctrl+Enter**
- See instant result: SPAM or HAM
- View confidence percentage
- See which trigger words were detected
- Try sample emails with one click

---

## 🔮 Future Improvements

- Integrate BERT/LSTM deep learning models
- Connect to Gmail API for live filtering
- Handle adversarial misspellings (fr3e, w!n)
- Multi-language spam detection
- Deploy to cloud (AWS / Heroku)

---

## 👨‍💻 Author

Built as a Cybersecurity project demonstrating practical application of Machine Learning in email threat detection.