# Research Notes & Method Selection

## 1. NLP Task Definition

**Primary Task:** Binary sentiment classification (positive / negative) on Twitter data  
**Secondary Task:** Abstractive summarization of classified tweet clusters  
**Dataset:** Sentiment140 — 1.6M labeled tweets

---

## 2. Literature Review Summary

### Sentiment Classification

| Model | Approach | Reported F1 on Twitter Data |
|-------|----------|-----------------------------|
| Logistic Regression + TF-IDF | Bag-of-words baseline | ~0.77 |
| LSTM / BiLSTM | Sequential RNN with embeddings | ~0.83 |
| BERT (fine-tuned) | Transformer + pre-training | ~0.92 |
| RoBERTa | Optimized BERT pre-training | ~0.93 |
| DistilBERT | Distilled BERT (faster) | ~0.90 |

**Selected:** BERT (`bert-base-uncased`) — best balance of accuracy and training stability, widely cited for sentiment tasks (Devlin et al., 2019).

**Why not RoBERTa?** RoBERTa offers marginal gains (~1%) but requires significantly more compute. Given project constraints, BERT is sufficient to meet our F1 ≥ 0.90 target.

**Why not DistilBERT?** DistilBERT is faster but slightly less accurate. We prioritize accuracy over inference speed for this project.

### Summarization

| Model | Approach | ROUGE-L Benchmark |
|-------|----------|-------------------|
| BART | Denoising seq2seq pre-training | ~0.44 (CNN/DM) |
| T5 | Text-to-text transfer | ~0.43 |
| GPT-2 | Autoregressive LM | ~0.35 |
| Pegasus | Gap-sentence pre-training | ~0.45 |

**Selected:** BART (`facebook/bart-large-cnn`) — pre-trained on CNN/DailyMail news summarization, strong abstractive generation, well-supported in HuggingFace (Lewis et al., 2020).

**Why not T5?** T5 requires prompt engineering ("summarize: ...") which adds complexity. BART is more straightforward for this task.

**Why not Pegasus?** Pegasus excels at news but is less tested on social media text.

---

## 3. Framework Selection

| Framework | Reason for Selection |
|-----------|----------------------|
| PyTorch | Primary deep learning framework; flexible and well-documented |
| HuggingFace Transformers | Industry-standard for BERT/BART; pre-trained models readily available |
| scikit-learn | Logistic Regression baseline; fast, reliable |
| NLTK | Tokenization, stop-word removal |
| rouge-score | ROUGE evaluation for summarization |

---

## 4. Benchmarking Results (Actual)

Final results on held-out test sets (see notebooks 02–05):

| Model | Train size | Accuracy | Macro F1 | Target | Met? |
|-------|-----------|----------|----------|--------|------|
| Logistic Regression | 200k | 0.81 | 0.805 | 0.77 | Yes |
| BiLSTM (GloVe) | 800k | 0.80 | 0.798 | 0.83 | No |
| BERT (fine-tuned) | 1.6M | 0.88 | 0.878 | 0.90 | No |
| BART (summarization) | — | — | ROUGE-L 0.17 | 0.40 | No |

Note: models were trained on different dataset sizes due to compute constraints,
so the table reflects each model's best achieved result rather than a strictly
controlled comparison. The LR baseline is competitive with the BiLSTM despite far
less complexity; BERT is clearly strongest but fell short of the 0.90 target.

---

## 5. Preprocessing Decisions

| Step | Decision | Rationale |
|------|----------|-----------|
| Lowercasing | Yes | Reduces vocabulary size |
| URL removal | Yes | URLs add noise, no sentiment value |
| @mention removal | Yes | Usernames carry no sentiment |
| Hashtag removal | Yes (symbol only, keep word) | Hashtag words can carry sentiment |
| Stop-word removal | Yes (for LR/LSTM) | Reduces noise for classical models |
| Stop-word removal | No (for BERT) | BERT uses full context including stop words |
| Stemming | No | Lemmatization preferred; BERT handles morphology |

---

## 6. Hyperparameter Choices

### Logistic Regression
- `C=1.0` (regularization), `max_iter=1000`, `solver='lbfgs'`
- TF-IDF: `max_features=50000`, `ngram_range=(1,2)`

### BiLSTM
- Embedding dim: 128, Hidden dim: 256, Layers: 2, Bidirectional: True
- Dropout: 0.3, Batch size: 256, LR: 1e-3, Epochs: 10

### BERT
- Base model: `bert-base-uncased`, Max length: 64 tokens
- LR: 2e-5 (AdamW), Warmup: 10% of steps, Epochs: 3 planned (2 actually run due to compute limits), Batch size: 32

---

## 7. References

- Devlin, J. et al. (2019). BERT. NAACL-HLT.
- Lewis, M. et al. (2020). BART. ACL.
- Go, A. et al. (2009). Sentiment140. Stanford.
- Liu, B. (2012). Sentiment Analysis and Opinion Mining.
