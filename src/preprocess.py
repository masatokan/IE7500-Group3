"""
preprocess.py
-------------
Preprocessing utilities for the Sentiment140 dataset.
Handles text cleaning, tokenization, stop-word removal, and normalization.
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

STOPWORDS = set(stopwords.words('english'))


def load_dataset(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load the Sentiment140 CSV and return a clean DataFrame.

    Args:
        filepath: Path to training.1600000.processed.noemoticon.csv
        sample_size: Optional number of rows to sample (for quick testing)

    Returns:
        DataFrame with columns ['label', 'text']
    """
    cols = ['polarity', 'id', 'date', 'query', 'user', 'text']
    df = pd.read_csv(filepath, encoding='latin-1', header=None, names=cols)
    df = df[['polarity', 'text']].copy()
    # Recode labels: 0 = negative, 4 -> 1 = positive
    df['label'] = df['polarity'].map({0: 0, 4: 1})
    df = df[['label', 'text']].dropna()
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """
    Clean a single tweet string.

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove @mentions and #hashtags
        4. Remove punctuation and numbers
        5. Strip extra whitespace

    Args:
        text: Raw tweet string

    Returns:
        Cleaned string
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # Remove URLs
    text = re.sub(r'@\w+', '', text)                     # Remove mentions
    text = re.sub(r'#\w+', '', text)                     # Remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)                 # Remove non-alpha
    text = re.sub(r'\s+', ' ', text).strip()             # Normalize whitespace
    return text


def tokenize_and_filter(text: str) -> list:
    """
    Tokenize text and remove stop words.

    Args:
        text: Cleaned tweet string

    Returns:
        List of filtered tokens
    """
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame.

    Args:
        df: DataFrame with 'text' column

    Returns:
        DataFrame with added 'clean_text' and 'tokens' columns
    """
    df = df.copy()
    df['clean_text'] = df['text'].apply(clean_text)
    df['tokens'] = df['clean_text'].apply(tokenize_and_filter)
    # Remove empty rows after cleaning
    df = df[df['tokens'].map(len) > 0].reset_index(drop=True)
    return df


if __name__ == '__main__':
    # Quick test
    sample = "Oh great, another update that broke everything! http://example.com @user #fail"
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))
    print("Tokens:  ", tokenize_and_filter(clean_text(sample)))
