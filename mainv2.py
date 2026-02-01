import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------------------------
# Global Config
# ------------------------------------------------------------------------------

SEED = 42
DATA_PATH = Path("data/SQLiV3.csv")

np.random.seed(SEED)
nltk.download("stopwords", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data Loading & Cleaning
# ------------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    logger.info("Loading dataset...")
    df = pd.read_csv(path)

    df = df[["Sentence", "Label"]].dropna()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df[df["Label"].isin([0, 1])]
    df["Label"] = df["Label"].astype(int)

    logger.info(f"Loaded {len(df)} samples")
    return df


def clean_text(text: str, stop_words: set) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", "", str(text).lower())
    return " ".join(w for w in text.split() if w not in stop_words)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing text...")
    stop_words = set(stopwords.words("english"))

    df = df.drop_duplicates(subset=["Sentence"])
    df["cleaned"] = df["Sentence"].apply(lambda x: clean_text(x, stop_words))

    return df


# ------------------------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------------------------

def vectorize_tfidf(X_train, X_test):
    logger.info("TF-IDF vectorization...")
    vectorizer = TfidfVectorizer()
    return (
        vectorizer.fit_transform(X_train),
        vectorizer.transform(X_test),
        vectorizer,
    )


def chi_square_select(X, y, k):
    selector = SelectKBest(chi2, k=k)
    return selector.fit_transform(X, y), selector


# ------------------------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------------------------

def evaluate(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    scores = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "FPR": [],
    }

    for tr, val in skf.split(X, y):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)

        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()

        scores["Accuracy"].append(accuracy_score(y_val, pred))
        scores["Precision"].append(precision_score(y_val, pred))
        scores["Recall"].append(recall_score(y_val, pred))
        scores["F1"].append(f1_score(y_val, pred))
        scores["FPR"].append(fp / (fp + tn))

    return {k: np.mean(v) for k, v in scores.items()}


# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

def plot_bar(results, title, filename):
    df = pd.DataFrame(results).T
    df.plot(kind="bar", figsize=(14, 8))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_tsne(X_before, X_after, y, sample=1000):
    tsne = TSNE(n_components=2, random_state=SEED)

    Xb = X_before[:sample].toarray()
    Xa = X_after[:sample].toarray()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].set_title("Before FS")
    sns.scatterplot(x=tsne.fit_transform(Xb)[:, 0],
                    y=tsne.fit_transform(Xb)[:, 1],
                    hue=y[:sample],
                    ax=ax[0])

    ax[1].set_title("After FS")
    sns.scatterplot(x=tsne.fit_transform(Xa)[:, 0],
                    y=tsne.fit_transform(Xa)[:, 1],
                    hue=y[:sample],
                    ax=ax[1])

    plt.show()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    df = preprocess(load_dataset(DATA_PATH))

    X = df["cleaned"]
    y = df["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    X_train_tf, X_test_tf, _ = vectorize_tfidf(X_train, X_test)

    # fixed k = 500 (đủ tốt, khỏi search lâu)
    X_train_fs, selector = chi_square_select(X_train_tf, y_train, k=500)

    models = {
        "MNB": MultinomialNB(),
        "LR": LogisticRegression(max_iter=1000),
        "DT": DecisionTreeClassifier(),
        "SVM": LinearSVC(),
        "KNN": KNeighborsClassifier(),
    }

    results_before = {name: evaluate(m, X_train_tf, y_train) for name, m in models.items()}
    results_after = {name: evaluate(m, X_train_fs, y_train) for name, m in models.items()}

    print("\nBefore FS:", results_before)
    print("\nAfter FS:", results_after)

    plot_bar(results_before, "Before Chi-square", "before.png")
    plot_bar(results_after, "After Chi-square", "after.png")

    plot_tsne(X_train_tf, X_train_fs, y_train)


if __name__ == "__main__":
    main()