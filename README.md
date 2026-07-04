# IE 7500 Applied NLP: Sentiment140 Classification & Analysis

## Overview
This project develops a two-stage NLP pipeline classifying sentiment and summarizing tweet clusters. Our objective is to move beyond simple labels by extracting actionable thematic insights from social media data.

## Research and Selection of Methods
We evaluated models based on their ability to handle the "noisy" nature of social media (ambiguity, sarcasm, OOV words)[cite: 1].
* **BiLSTM (Baseline):** Chosen for sequential context. We injected 100-dimensional pre-trained **GloVe** embeddings to compensate for vocabulary sparsity[cite: 2].
* **BERT (State-of-the-Art):** A transformer-based architecture utilizing multi-head self-attention for deep semantic context[cite: 1].
* **BART (Generative Summarization):** Implemented to synthesize thematic insights from tweet clusters[cite: 1].

## Implementation Details
* **Frameworks:** Models built using **PyTorch** and **Hugging Face**[cite: 1].
* **Dataset:** 1.6M tweets from the **Sentiment140** benchmark[cite: 1].
* **Optimization:** BERT training was scaled on an **A100 High-RAM GPU**, leveraging explicit CUDA memory management to handle dataset volume[cite: 2].

## Evaluation Results

| Model | Accuracy | Macro F1-Score | ROC-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: |
| **BiLSTM (GloVe)** | 80.0% | 0.7977 | 0.8817 | 0.5954 |
| **BERT (Uncased)** | 88.0% | 0.8784 | 0.9492 | 0.7568 |

## Team Roles
| Member | Role |
|---|---|
| Abdellah Faleh | Project coordination, proposal writing, NLP preprocessing[cite: 1] |
| Masato Kan | Data collection, dataset preparation, feature engineering[cite: 1] |
| Ambreen Khan | Model evaluation, visualizations, ROUGE scoring, documentation[cite: 1] |
| Carlo Porras | Model implementation (LR, LSTM, BERT) and BART integration[cite: 1] |

## References
Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers.
Lewis et al. (2020). BART: Denoising sequence-to-sequence pre-training.
Go et al. (2009). Twitter sentiment classification using distant supervision.
Sokolova & Lapalme (2009). A systematic analysis of performance measures for classification tasks.