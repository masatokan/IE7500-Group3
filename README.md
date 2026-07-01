# Sentiment Analysis Using NLP and Generative AI

### A Transformer-Based Classification and Summarization Pipeline

**Group 03** | Abdellah Faleh · Masato Kan · Ambreen Khan · Carlo Porras

---

## 📌 Project Overview

This project builds a two-stage NLP pipeline that:

- Classifies tweets as positive or negative using three models of increasing complexity

- Summarizes grouped tweet clusters using BART generative AI

We use the **Sentiment140 dataset** — 1.6 million labeled tweets from Twitter.

---

## 🗂️ Repository Structure

nlp_project/

├── README.md

├── data/

│ └── training.1600000.processed.noemoticon.csv

├── notebooks/

│ ├── 02_logistic_regression.ipynb

│ ├── 03_lstm_model.ipynb

│ ├── 04_bert_model.ipynb

│ └── 05_bart_summarization.ipynb

├── src/

│ ├── preprocess.py

│ └── evaluate.py

├── models/

├── results/

└── docs/

└── research_notes.md

---

## 🧠 Models and Results

| Model | Target | Achieved | Status |

| Logistic Regression | F1 ≥ 0.77 | F1 = 0.805, Accuracy = 81%, ROC-AUC = 0.885 | ✅ Target met |

| BiLSTM | F1 ≥ 0.83 | F1 = 0.7705, Accuracy = 77%, ROC-AUC = 0.853 | ⚠️ Below target |

| BERT (fine-tuned) | F1 ≥ 0.90 | In progress | ⏳ Pending |

| BART (summarization) | ROUGE-L ≥ 0.40 | ROUGE-L = 0.06 | ⚠️ Below target |

Note on BiLSTM: Trained on CPU for 5 epochs. Loss decreased steadily (0.52 to 0.45), suggesting further improvement is possible with GPU access or more epochs.

Note on BART: Applied zero-shot using facebook/bart-large-cnn. Low ROUGE-L reflects domain mismatch between BART pretraining on news articles and informal Twitter text. See research_notes.md for full discussion.

---

## ⚙️ Setup Instructions

1. Clone the repository

git clone https://github.com/your-username/nlp_project.git

cd nlp_project

2. Install dependencies

pip install -r requirements.txt

3. Download the dataset

Download training.1600000.processed.noemoticon.csv from:

https://www.kaggle.com/datasets/kazanova/sentiment140

Place the CSV file in the data/ folder.

4. Run notebooks in order

02_logistic_regression.ipynb

03_lstm_model.ipynb

04_bert_model.ipynb

05_bart_summarization.ipynb

Note: preprocess.py and evaluate.py must be in the same folder as the notebooks when running locally.

---

## 📦 Requirements

pandas

numpy

scikit-learn

nltk

torch

transformers

datasets

rouge-score

matplotlib

seaborn

jupyter

joblib

---

## 📊 Evaluation Metrics

Primary: Macro-averaged F1-score

Secondary: Accuracy, Precision, Recall, ROC-AUC, MCC

Summarization: ROUGE-L, human Likert scale (1-5)

---

## 👥 Team Roles

| Member | Role |

| Abdellah Faleh | Project coordination, proposal writing, NLP preprocessing |

| Masato Kan | Data collection, dataset preparation, feature engineering |

| Ambreen Khan | Model evaluation, visualizations, ROUGE scoring, documentation |

| Carlo Porras | Model implementation (LR, LSTM, BERT) and BART integration |

---

## 📚 References

Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers.

Lewis et al. (2020). BART: Denoising sequence-to-sequence pre-training.

Go et al. (2009). Twitter sentiment classification using distant supervision.

Sokolova and Lapalme (2009). A systematic analysis of performance measures for classification tasks.
