import os
import glob
import re
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import joblib
import json
from datetime import datetime


DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
PFNC_DIR = os.path.join(DATA_ROOT, 'Philippine-Fake-News-Corpus')
# Accept either .csv or .txt for slang words (repo currently has CSV)
SLANG_TXT_PATH = os.path.join(DATA_ROOT, 'filipino_slang_words.txt')
SLANG_CSV_PATH = os.path.join(DATA_ROOT, 'filipino_slang_words.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


def _find_pfnc_csv() -> Optional[str]:
    """Find a CSV file inside the Philippine-Fake-News-Corpus directory."""
    pattern = os.path.join(PFNC_DIR, '**', '*.csv')
    files = glob.glob(pattern, recursive=True)
    return files[0] if files else None


def _find_top_level_corpus_csv() -> Optional[str]:
    """Find the top-level 'Philippine Fake News Corpus.csv' file under data/."""
    candidate = os.path.join(DATA_ROOT, 'Philippine Fake News Corpus.csv')
    return candidate if os.path.exists(candidate) else None


def _select_text_and_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Heuristically pick text and label columns."""
    text_candidates = ['text', 'content', 'body', 'article', 'headline']
    label_candidates = ['label', 'category', 'target', 'is_fake', 'verdict', 'class']

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    # Fallbacks
    if text_col is None:
        # Pick the longest average-length string column
        str_cols = [c for c in df.columns if df[c].dtype == 'object']
        if str_cols:
            text_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
    if label_col is None:
        # Pick a low-cardinality column
        label_col = min(df.columns, key=lambda c: df[c].nunique())

    if text_col is None or label_col is None:
        raise ValueError('Unable to determine text/label columns from dataset.')

    return text_col, label_col


def _normalize_labels(series: pd.Series) -> pd.Series:
    """Map labels to {'fake', 'real'} if possible, else keep as-is."""
    s = series.astype(str).str.strip().str.lower()
    unique = set(s.unique())

    mapping = None
    if {'fake', 'real'}.issubset(unique):
        mapping = {'fake': 'fake', 'real': 'real'}
    elif {'true', 'false'}.issubset(unique):
        mapping = {'false': 'fake', 'true': 'real'}
    elif {'1', '0'}.issubset(unique):
        mapping = {'1': 'fake', '0': 'real'}

    return s.map(mapping) if mapping else s


def load_pfnc_dataset() -> Optional[pd.DataFrame]:
    """Load the Philippine Fake News Corpus dataset as a DataFrame with 'text' and 'label'."""
    csv_path = _find_pfnc_csv()
    if not csv_path:
        print(f"PFNC CSV not found under: {PFNC_DIR}")
        return None

    # Read robustly: use Python open() to handle decoding errors, then pass file handle to pandas
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            df = pd.read_csv(f)
    except Exception:
        # Fallback to default pandas handling if the above fails
        df = pd.read_csv(csv_path, encoding='latin-1')
    text_col, label_col = _select_text_and_label_columns(df)
    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
    df['label'] = _normalize_labels(df['label'])

    # Drop rows with missing values
    df = df.dropna(subset=['text', 'label'])
    print(f"Loaded PFNC: {csv_path} | rows={len(df)} | columns={list(df.columns)}")
    print(df['label'].value_counts())
    return df


def load_top_level_corpus() -> Optional[pd.DataFrame]:
    """Load the top-level 'Philippine Fake News Corpus.csv' if present."""
    csv_path = _find_top_level_corpus_csv()
    if not csv_path:
        print("Top-level 'Philippine Fake News Corpus.csv' not found under data/.")
        return None

    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            df = pd.read_csv(f)
    except Exception:
        df = pd.read_csv(csv_path, encoding='latin-1')

    text_col, label_col = _select_text_and_label_columns(df)
    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
    df['label'] = _normalize_labels(df['label'])
    df = df.dropna(subset=['text', 'label'])
    print(f"Loaded Top-level Corpus: {csv_path} | rows={len(df)} | columns={list(df.columns)}")
    print(df['label'].value_counts())
    return df


def visualize_label_distribution(labels: pd.Series, title: str):
    ax = sns.countplot(x=labels)
    ax.set_title(title)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def visualize_confusion_matrix(y_true, y_pred, labels: List[str], title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    ax = sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()


def train_random_forest_on_pfnc():
    """Train a Random Forest news credibility classifier on the PFNC dataset."""
    df = load_pfnc_dataset()
    if df is None:
        return

    # Basic visualization of class distribution
    visualize_label_distribution(df['label'], 'PFNC Class Distribution')

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].astype(str), df['label'].astype(str), test_size=0.2, random_state=42, stratify=df['label']
    )

    # Vectorize -> Reduce dimension -> Random Forest
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)
    svd = TruncatedSVD(n_components=300, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')

    pipeline = make_pipeline(tfidf, svd, rf)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print('\nRandom Forest on PFNC — Classification Report')
    print(classification_report(y_test, y_pred))

    visualize_confusion_matrix(y_test, y_pred, labels=sorted(df['label'].unique()),
                               title='PFNC Random Forest Confusion Matrix')

    # Persist trained pipeline and metadata for integration
    os.makedirs(MODEL_DIR, exist_ok=True)
    rf_model_path = os.path.join(MODEL_DIR, 'news_rf_pipeline.joblib')
    joblib.dump(pipeline, rf_model_path)
    print(f"Saved Random Forest pipeline to: {rf_model_path}")

    rf_meta = {
        'model': 'RandomForestClassifier',
        'pipeline': ['TfidfVectorizer(1-2gram)', 'TruncatedSVD(300)', 'RandomForest(300)'],
        'classes': sorted(df['label'].unique()),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'dataset': 'Philippine-Fake-News-Corpus',
        'vectorizer': {
            'type': 'tfidf',
            'ngram_range': [1, 2],
            'max_features': 10000,
            'min_df': 2
        }
    }
    with open(os.path.join(MODEL_DIR, 'news_rf_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(rf_meta, f, ensure_ascii=False, indent=2)


def train_combined_news_model():
    """Train TF-IDF + Logistic Regression on available corpora.
    Prefer the PFNC dataset as the initial dataset for compatibility.
    Persist pipeline and metrics: Accuracy, Precision, Recall, F1.
    """
    df1 = load_pfnc_dataset()
    df2 = load_top_level_corpus()

    if df1 is None and df2 is None:
        print('No news datasets found under data/. Skipping model training.')
        return

    # Prefer PFNC-only when available; otherwise fall back to top-level corpus
    if df1 is not None:
        df = df1.copy().reset_index(drop=True)
        dataset_sources = ['PFNC']
        cm_title = 'PFNC Logistic Regression Confusion Matrix'
        dist_title = 'PFNC Class Distribution'
        print(f"Using PFNC dataset only for training. rows={len(df)}")
    else:
        df = df2.copy().reset_index(drop=True)
        dataset_sources = ['TopLevel']
        cm_title = 'Top-Level Logistic Regression Confusion Matrix'
        dist_title = 'Top-Level Class Distribution'
        print(f"PFNC not found. Using Top-level corpus only. rows={len(df)}")

    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str)
    print(df['label'].value_counts())

    visualize_label_distribution(df['label'], dist_title)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42,
        stratify=df['label'] if df['label'].nunique() > 1 else None
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
    lr = LogisticRegression(max_iter=2000, solver='saga', class_weight='balanced')
    pipeline = make_pipeline(tfidf, lr)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # Compute additional metrics via classification_report
    report = classification_report(y_test, y_pred, output_dict=True)
    precision_macro = float(report.get('macro avg', {}).get('precision', 0.0))
    recall_macro = float(report.get('macro avg', {}).get('recall', 0.0))
    f1_macro = float(report.get('macro avg', {}).get('f1-score', 0.0))

    print('\nLogistic Regression — Classification Report')
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.4f} | Precision: {precision_macro:.4f} | Recall: {recall_macro:.4f} | F1: {f1_macro:.4f}")

    visualize_confusion_matrix(y_test, y_pred, labels=sorted(df['label'].unique()), title=cm_title)

    os.makedirs(MODEL_DIR, exist_ok=True)
    lr_model_path = os.path.join(MODEL_DIR, 'news_lr_pipeline.joblib')
    joblib.dump(pipeline, lr_model_path)
    print(f"Saved Logistic Regression pipeline to: {lr_model_path}")

    metrics = {
        'model': 'LogisticRegression',
        'pipeline': ['TfidfVectorizer(1-2gram,max_features=20000,min_df=2)', 'LogisticRegression(saga,balanced)'],
        'classes': sorted(df['label'].unique()),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'dataset': {
            'sources': dataset_sources,
            'rows': int(len(df))
        },
        'metrics': {
            'accuracy': float(acc),
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        }
    }
    with open(os.path.join(MODEL_DIR, 'news_lr_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_saved_news_model(model_filename: str = 'news_lr_pipeline.joblib'):
    """Load a saved news classification pipeline from models/."""
    path = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(path):
        print(f"Model file not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")
        return None


def predict_news_label(text: str, model=None) -> Optional[str]:
    """Predict 'fake' vs 'real' label for a news text using the saved pipeline."""
    if model is None:
        model = load_saved_news_model()
    if model is None:
        return None
    try:
        return str(model.predict([str(text)])[0])
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None


def _tokenize_words(text: str) -> List[str]:
    # Basic tokenizer for Filipino/English words
    return [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-']+", text)]


def load_slang_words() -> List[str]:
    # Prefer CSV if available
    if os.path.exists(SLANG_CSV_PATH):
        try:
            df = pd.read_csv(SLANG_CSV_PATH)
            # Heuristically choose a column containing words
            word_col = next((c for c in df.columns if 'word' in c.lower() or 'slang' in c.lower()), df.columns[0])
            words = [str(w).strip().lower() for w in df[word_col].tolist() if str(w).strip()]
            print(f"Loaded slang words: {len(words)} from {SLANG_CSV_PATH}")
            return words
        except Exception as e:
            print(f"Failed to read slang CSV at {SLANG_CSV_PATH}: {e}")
    # Fallback to TXT if exists
    if os.path.exists(SLANG_TXT_PATH):
        with open(SLANG_TXT_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            words = [w.strip().lower() for w in f if w.strip()]
            print(f"Loaded slang words: {len(words)} from {SLANG_TXT_PATH}")
            return words
    # Fallback: small placeholder list (replace with your dataset)
    fallback = ['werpa', 'lodi', 'petmalu', 'sanaol', 'charot', 'jowa']
    print(f"Slang file not found at {SLANG_TXT_PATH} or {SLANG_CSV_PATH}. Using fallback list: {fallback}")
    return fallback


def build_slang_dataset_from_corpus(slang_words: List[str], df_pfnc: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Create a token-level dataset: slang vs non_slang.
    Negative samples are drawn from PFNC corpus tokens not in slang list.
    """
    slang_words = list({w.lower() for w in slang_words})
    pos = pd.DataFrame({'word': slang_words, 'label': 'slang'})

    neg_words: List[str] = []
    if df_pfnc is not None:
        for txt in df_pfnc['text'].astype(str).tolist():
            for tok in _tokenize_words(txt):
                if tok not in slang_words and tok.isalpha() and len(tok) >= 3:
                    neg_words.append(tok)
        # Deduplicate and sample to balance
        neg_words = list({w for w in neg_words})
        random.seed(42)
        random.shuffle(neg_words)
        neg_words = neg_words[:len(slang_words)] if neg_words else []

    # Fallback negatives if corpus is missing/too small
    if not neg_words:
        neg_words = ['bahay', 'trabaho', 'balita', 'guro', 'paaralan', 'kape'][:len(slang_words)]

    neg = pd.DataFrame({'word': neg_words, 'label': 'non_slang'})
    ds = pd.concat([pos, neg], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Slang dataset built — rows={len(ds)} | slang={len(pos)} | non_slang={len(neg)}")
    return ds


def train_naive_bayes_for_slang():
    """Train a Naive Bayes classifier to detect Filipino slang words."""
    df_pfnc = load_pfnc_dataset()
    slang_words = load_slang_words()
    ds = build_slang_dataset_from_corpus(slang_words, df_pfnc)

    visualize_label_distribution(ds['label'], 'Slang Dataset Class Distribution')

    X_train, X_test, y_train, y_test = train_test_split(
        ds['word'].astype(str), ds['label'].astype(str), test_size=0.2, random_state=42, stratify=ds['label']
    )

    # Character n-grams often work well for short tokens
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
    nb = MultinomialNB()
    pipeline = make_pipeline(vec, nb)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print('\nNaive Bayes for Slang — Classification Report')
    print(classification_report(y_test, y_pred))

    visualize_confusion_matrix(y_test, y_pred, labels=sorted(ds['label'].unique()),
                               title='Slang Naive Bayes Confusion Matrix')

    # Show top char-ngrams per class (interpretability)
    vec_fit = vec.fit(X_train)
    X_train_vec = vec_fit.transform(X_train)
    nb_fit = MultinomialNB().fit(X_train_vec, y_train)
    feature_names = np.array(vec_fit.get_feature_names_out())
    for i, cls in enumerate(nb_fit.classes_):
        top_idx = np.argsort(nb_fit.class_log_prior_[i] + nb_fit.feature_log_prob_[i])[-15:]
        print(f"Top n-grams for class={cls}: {feature_names[top_idx]}")

    # Persist trained pipeline and metadata for integration
    os.makedirs(MODEL_DIR, exist_ok=True)
    slang_model_path = os.path.join(MODEL_DIR, 'slang_nb_pipeline.joblib')
    joblib.dump(pipeline, slang_model_path)
    print(f"Saved Slang Naive Bayes pipeline to: {slang_model_path}")

    slang_meta = {
        'model': 'MultinomialNB',
        'pipeline': ['TfidfVectorizer(char 3-5gram)', 'MultinomialNB'],
        'classes': sorted(ds['label'].unique()),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'dataset': 'filipino_slang_words (tokens + negatives from PFNC)',
        'vectorizer': {
            'type': 'tfidf-char',
            'ngram_range': [3, 5]
        }
    }
    with open(os.path.join(MODEL_DIR, 'slang_nb_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(slang_meta, f, ensure_ascii=False, indent=2)


def main():
    sns.set_theme(style='whitegrid')
    print('--- Training Combined News Model (TF-IDF + Logistic) ---')
    train_combined_news_model()
    print('--- Training Random Forest on PFNC ---')
    train_random_forest_on_pfnc()

    print('\n--- Training Naive Bayes for Filipino Slang ---')
    train_naive_bayes_for_slang()


if __name__ == '__main__':
    main()