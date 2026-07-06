# IE 7500 Applied NLP: Sentiment140 Classification & Analysis

## Overview
This project develops a two-stage NLP pipeline classifying sentiment and summarizing tweet clusters. Our objective is to move beyond simple labels by extracting actionable thematic insights from social media data.

## Research and Selection of Methods
We evaluated models based on their ability to handle the "noisy" nature of social media (ambiguity, sarcasm, OOV words).
* **Logistic Regression (Baseline):** TF-IDF (1–2 grams) baseline; fast, interpretable, and a strong reference point for short-text sentiment.
* **BiLSTM:** Chosen for sequential context. We injected 100-dimensional pre-trained **GloVe** embeddings to compensate for vocabulary sparsity.
* **BERT (State-of-the-Art):** A transformer-based architecture utilizing multi-head self-attention for deep semantic context.
* **BART (Generative Summarization):** Implemented to synthesize thematic insights from tweet clusters.

## Implementation Details
* **Frameworks:** Models built using **PyTorch** and **Hugging Face**.
* **Dataset:** 1.6M tweets from the **Sentiment140** benchmark.
* **Optimization:** BERT training was scaled on an **A100 High-RAM GPU**, leveraging explicit CUDA memory management to handle dataset volume.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/masatokan/IE7500-Group3.git
   cd IE7500-Group3
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the dataset:** get the Sentiment140 CSV from Kaggle
   (https://www.kaggle.com/datasets/kazanova/sentiment140) and place
   `training.1600000.processed.noemoticon.csv` in the `data/` folder. The file is
   not committed to the repo due to its size, so this step is required before
   running any notebook.
4. **Run the notebooks** in order (`notebooks/02` → `05`). Trained models are
   written to `models/` and figures/metrics to `results/`.

### A note on the notebooks

Notebooks `03` (BiLSTM) and `04` (BERT) are provided in **two versions**:

* `03_lstm_model.ipynb` / `04_bert_model.ipynb` — local versions.
* `03_lstm_model_colab.ipynb` / `04_bert_model_colab.ipynb` — Google Colab versions.

Both versions implement the same models. We keep both to allow experimentation
across development environments: these models are compute-intensive, and the Colab
versions were used to train on a GPU (A100/L4) at full data scale, while the local
versions support running in a standard Python/Jupyter environment.

## Model Comparison

| Model              | Train size | Accuracy | Macro F1 | ROC-AUC | MCC    | Target | Met? |
| ------------------ | ---------- | -------- | -------- | ------- | ------ | ------ | ---- |
| Logistic Regression| 200k       | 0.81     | 0.805    | 0.885   | 0.61   | 0.77   | Yes  |
| BiLSTM (GloVe)     | 800k       | 0.80     | 0.798    | 0.882   | 0.595  | 0.83   | No   |
| BERT (uncased)     | 1.6M       | 0.88     | 0.878    | 0.949   | 0.757  | 0.90   | No   |

Summarization (BART):

| Metric   | Score  | Target      | Met? |
| -------- | ------ | ----------- | ---- |
| ROUGE-1  | 0.1915 | —           | —    |
| ROUGE-2  | 0.0417 | —           | —    |
| ROUGE-L  | 0.1715 | ≥ 0.40      | No   |

**Note on comparability:** the models were trained on different dataset sizes
(200k / 800k / 1.6M) due to compute constraints, so this table shows each model's
best achieved result rather than a strictly controlled comparison. The main
qualitative finding holds regardless: BERT is clearly strongest, while the
LR baseline is competitive with the BiLSTM despite far less complexity.

The summarization component did not meet its ROUGE-L target. A news-trained
summarizer transfers poorly to noisy, non-narrative tweet clusters: the model
tended to copy fragments rather than synthesize themes. We report this as a
negative result and outline topic-clustering and domain-appropriate models as
next steps (see `notebooks/05_bart_summarization.ipynb`).

## Team Roles
| Member | Role |
|---|---|
| Abdellah Faleh | Project coordination, proposal writing, NLP preprocessing |
| Masato Kan | Data collection, dataset preparation, feature engineering |
| Ambreen Khan | Model evaluation, visualizations, ROUGE scoring, documentation |
| Carlo Porras | Model implementation (LR, LSTM, BERT) and BART integration |

## References
Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers.
Lewis et al. (2020). BART: Denoising sequence-to-sequence pre-training.
Go et al. (2009). Twitter sentiment classification using distant supervision.
Sokolova & Lapalme (2009). A systematic analysis of performance measures for classification tasks.
