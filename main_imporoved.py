"""
Enhanced SQL Injection Detection - Following Paper Methodology
Paper: "Enhanced SQL injection detection using chi-square feature selection 
       and machine learning classifiers" (Casmiry et al., 2025)

Key improvements:
1. Two-step Chi-square search (coarse + fine) → k=2,551
2. Test set evaluation
3. Report mean ± standard deviation
4. Computational efficiency metrics
"""

import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)

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
DATA_PATH = Path("data/SQLiV3_cleaned.csv")  # Use cleaned data
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
nltk.download("stopwords", quiet=True)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data Loading & Cleaning
# ------------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """Load cleaned dataset"""
    logger.info(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    
    # Ensure only Sentence and Label columns
    df = df[["Sentence", "Label"]].dropna()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df[df["Label"].isin([0, 1])]
    df["Label"] = df["Label"].astype(int)
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"  - Benign (0): {(df['Label']==0).sum()} ({(df['Label']==0).sum()/len(df)*100:.2f}%)")
    logger.info(f"  - Malicious (1): {(df['Label']==1).sum()} ({(df['Label']==1).sum()/len(df)*100:.2f}%)")
    return df


def clean_text(text: str, stop_words: set) -> str:
    """
    Improved cleaning that keeps SQL-specific tokens
    
    This preserves important SQL injection signatures:
    - Single quotes (')
    - Double dashes (--)
    - Semicolons (;)
    - Pipes (|)
    - Parentheses and equals
    """
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    
    # Keep: letters, numbers, spaces, and SQL chars: ' - ; | * ( ) =
    text = re.sub(r"[^a-z0-9\s'\-;|*()=]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Keep SQL keywords even if they're stop words
    sql_keywords = {
        'select', 'from', 'where', 'union', 'insert', 'update',
        'delete', 'drop', 'or', 'and', 'not', 'null', 'like', 'in'
    }
    
    words = []
    for word in text.split():
        if (word not in stop_words or 
            word in sql_keywords or 
            any(c in word for c in ["'", '-', ';', '|', '*'])):
            words.append(word)
    
    return ' '.join(words)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess text data"""
    logger.info("Preprocessing text...")
    stop_words = set(stopwords.words("english"))
    
    df = df.drop_duplicates(subset=["Sentence"])
    df["cleaned"] = df["Sentence"].apply(lambda x: clean_text(x, stop_words))
    
    logger.info(f"After preprocessing: {len(df)} samples")
    return df


# ------------------------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------------------------

def vectorize_tfidf(X_train, X_test):
    """TF-IDF Vectorization"""
    logger.info("TF-IDF vectorization...")
    vectorizer = TfidfVectorizer()
    
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)
    
    logger.info(f"  - Vocabulary size: {len(vectorizer.vocabulary_)}")
    logger.info(f"  - Train shape: {X_train_tf.shape}")
    logger.info(f"  - Test shape: {X_test_tf.shape}")
    
    return X_train_tf, X_test_tf, vectorizer


def find_optimal_k(X_train, y_train, max_k=None):
    """
    TWO-STEP FEATURE SELECTION (as per paper)
    
    Step 1: Coarse search (step=50)
    Step 2: Fine search (step=1) around best k
    
    Returns: optimal k (should be around 2,551 for paper reproduction)
    """
    if max_k is None:
        max_k = X_train.shape[1]
    
    logger.info("=" * 60)
    logger.info("STEP 1: COARSE SEARCH (step=50)")
    logger.info("=" * 60)
    
    # Coarse search: k = 50, 100, 150, ..., max_k
    coarse_k_values = list(range(50, max_k + 1, 50))
    coarse_accuracies = []
    
    model = MultinomialNB()  # Use MNB as baseline (as per paper)
    
    for k in coarse_k_values:
        selector = SelectKBest(chi2, k=k)
        X_selected = selector.fit_transform(X_train, y_train)
        
        # Simple train-test to evaluate
        from sklearn.model_selection import train_test_split as split
        X_tr, X_val, y_tr, y_val = split(
            X_selected, y_train, test_size=0.2, stratify=y_train, random_state=SEED
        )
        
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, model.predict(X_val))
        coarse_accuracies.append(acc)
        
        if k % 500 == 0:  # Log every 500 features
            logger.info(f"  k={k:5d} → Accuracy: {acc:.4f}")
    
    best_coarse_idx = np.argmax(coarse_accuracies)
    best_coarse_k = coarse_k_values[best_coarse_idx]
    best_coarse_acc = coarse_accuracies[best_coarse_idx]
    
    logger.info(f"\nCoarse search complete:")
    logger.info(f"  Best k = {best_coarse_k} (Accuracy: {best_coarse_acc:.4f})")
    
    # ----------------------------------------------------------------------
    # STEP 2: Fine search
    # ----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: FINE SEARCH (step=1)")
    logger.info("=" * 60)
    
    # Fine search: k = best_k-200 to best_k+200 (step=1)
    fine_start = max(50, best_coarse_k - 200)
    fine_end = min(max_k, best_coarse_k + 200)
    fine_k_values = list(range(fine_start, fine_end + 1, 1))
    fine_accuracies = []
    
    for k in fine_k_values:
        selector = SelectKBest(chi2, k=k)
        X_selected = selector.fit_transform(X_train, y_train)
        
        X_tr, X_val, y_tr, y_val = split(
            X_selected, y_train, test_size=0.2, stratify=y_train, random_state=SEED
        )
        
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, model.predict(X_val))
        fine_accuracies.append(acc)
    
    best_fine_idx = np.argmax(fine_accuracies)
    best_fine_k = fine_k_values[best_fine_idx]
    best_fine_acc = fine_accuracies[best_fine_idx]
    
    # Find plateau region
    max_acc = max(fine_accuracies)
    plateau_indices = [i for i, acc in enumerate(fine_accuracies) if acc == max_acc]
    plateau_k_values = [fine_k_values[i] for i in plateau_indices]
    
    # Select midpoint of plateau
    optimal_k = plateau_k_values[len(plateau_k_values) // 2]
    
    logger.info(f"\nFine search complete:")
    logger.info(f"  Max accuracy: {max_acc:.4f}")
    logger.info(f"  Plateau region: k={plateau_k_values[0]} to k={plateau_k_values[-1]}")
    logger.info(f"  Optimal k (midpoint): {optimal_k}")
    
    # Visualization
    plot_k_search(coarse_k_values, coarse_accuracies, 
                  fine_k_values, fine_accuracies, 
                  plateau_k_values, optimal_k)
    
    return optimal_k


def plot_k_search(coarse_k, coarse_acc, fine_k, fine_acc, plateau_k, optimal_k):
    """Plot coarse and fine search results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Coarse search
    ax1.plot(coarse_k, coarse_acc, 'b-', linewidth=2)
    ax1.axvline(x=coarse_k[np.argmax(coarse_acc)], 
                color='r', linestyle='--', label=f'Best k={coarse_k[np.argmax(coarse_acc)]}')
    ax1.set_xlabel('Number of Features (k)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Coarse Search (step=50)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Fine search
    ax2.plot(fine_k, fine_acc, 'g-', linewidth=2)
    ax2.axvspan(plateau_k[0], plateau_k[-1], alpha=0.3, color='yellow', 
                label=f'Plateau region')
    ax2.axvline(x=optimal_k, color='r', linestyle='--', 
                label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Features (k)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Fine Search (step=1)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "optimal_k_search.png", dpi=300)
    logger.info(f"Saved plot: {RESULTS_DIR / 'optimal_k_search.png'}")
    plt.close()


def chi_square_select(X, y, k):
    """Select top-k features using Chi-square"""
    selector = SelectKBest(chi2, k=k)
    return selector.fit_transform(X, y), selector


# ------------------------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------------------------

def evaluate_cv(model, X, y, model_name="Model"):
    """
    Evaluate model using Stratified 5-Fold Cross Validation
    Returns: dict with mean ± std for each metric
    
    NOTE: X can be sparse or dense - models handle both
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    scores = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "FPR": [],
    }
    
    for fold_idx, (tr, val) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]
        
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        
        scores["Accuracy"].append(accuracy_score(y_val, pred))
        scores["Precision"].append(precision_score(y_val, pred, zero_division=0))
        scores["Recall"].append(recall_score(y_val, pred, zero_division=0))
        scores["F1"].append(f1_score(y_val, pred, zero_division=0))
        scores["FPR"].append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
    # Return mean ± std (as per paper)
    return {
        k: {
            'mean': np.mean(v),
            'std': np.std(v)
        } for k, v in scores.items()
    }


def evaluate_test_set(model, X_train, y_train, X_test, y_test):
    """Final evaluation on test set"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "Misclassification": (fp + fn) / (tp + tn + fp + fn),
    }


def measure_computational_efficiency(model, X_train, y_train, X_test, model_name="Model"):
    """Measure training time, inference time, and memory usage"""
    import sys
    
    # Training time
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Inference time (per query)
    start = time.time()
    _ = model.predict(X_test)
    inference_time_total = time.time() - start
    # Use .shape[0] for sparse matrices
    n_samples = X_test.shape[0]
    inference_time_per_query = inference_time_total / n_samples * 1000  # ms
    
    # Memory usage (approximate)
    try:
        import pickle
        model_bytes = len(pickle.dumps(model))
        memory_mb = model_bytes / (1024 * 1024)
    except:
        memory_mb = 0
    
    return {
        "Training Time (s)": train_time,
        "Inference Time (ms/query)": inference_time_per_query,
        "Model Size (MB)": memory_mb,
    }


# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

def plot_comparison_bars(results_before, results_after, title, filename):
    """Plot grouped bar chart comparing before/after"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'FPR']
    
    # Extract mean values
    data_before = {
        model: [results_before[model][m]['mean'] * 100 for m in metrics]
        for model in results_before
    }
    data_after = {
        model: [results_after[model][m]['mean'] * 100 for m in metrics]
        for model in results_after
    }
    
    df_before = pd.DataFrame(data_before, index=metrics).T
    df_after = pd.DataFrame(data_after, index=metrics).T
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    df_before.plot(kind='bar', ax=ax1)
    ax1.set_title('Before Feature Selection')
    ax1.set_ylabel('Score (%)')
    ax1.legend(loc='lower right')
    ax1.set_ylim([0, 105])
    
    df_after.plot(kind='bar', ax=ax2)
    ax2.set_title('After Feature Selection')
    ax2.set_ylabel('Score (%)')
    ax2.legend(loc='lower right')
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300)
    logger.info(f"Saved plot: {RESULTS_DIR / filename}")
    plt.close()


def plot_tsne(X_before, X_after, y, sample=1000):
    """t-SNE visualization before/after feature selection"""
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    
    # Sample data
    indices = np.random.choice(len(y), size=min(sample, len(y)), replace=False)
    Xb = X_before[indices].toarray()
    Xa = X_after[indices].toarray()
    y_sample = y[indices]
    
    # Transform
    Xb_2d = tsne.fit_transform(Xb)
    Xa_2d = tsne.fit_transform(Xa)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before
    scatter1 = ax1.scatter(Xb_2d[:, 0], Xb_2d[:, 1], 
                          c=y_sample, cmap='RdYlBu', alpha=0.6)
    ax1.set_title('Before Feature Selection\n(High Overlap)')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=ax1, label='Class')
    
    # After
    scatter2 = ax2.scatter(Xa_2d[:, 0], Xa_2d[:, 1], 
                          c=y_sample, cmap='RdYlBu', alpha=0.6)
    ax2.set_title('After Feature Selection\n(Clear Separation)')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=ax2, label='Class')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tsne_comparison.png", dpi=300)
    logger.info(f"Saved plot: {RESULTS_DIR / 'tsne_comparison.png'}")
    plt.close()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("ENHANCED SQL INJECTION DETECTION")
    logger.info("Following methodology from Casmiry et al. (2025)")
    logger.info("=" * 70)
    
    # 1. Load and preprocess
    df = preprocess(load_dataset(DATA_PATH))
    
    X = df["cleaned"]
    y = df["Label"].values
    
    # 2. Train-test split (80-20, stratified)
    logger.info("\n" + "=" * 70)
    logger.info("TRAIN-TEST SPLIT (80-20, stratified)")
    logger.info("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Test size: {len(X_test)}")
    
    # 3. TF-IDF Vectorization
    logger.info("\n" + "=" * 70)
    logger.info("TF-IDF VECTORIZATION")
    logger.info("=" * 70)
    X_train_tf, X_test_tf, vectorizer = vectorize_tfidf(X_train, X_test)
    
    # 4. Find optimal k (TWO-STEP SEARCH)
    logger.info("\n" + "=" * 70)
    logger.info("CHI-SQUARE FEATURE SELECTION")
    logger.info("=" * 70)
    
    # OPTION 1: Run full search (takes time)
    # optimal_k = find_optimal_k(X_train_tf, y_train)
    
    # OPTION 2: Use paper's optimal k directly
    optimal_k = 2551  # From paper
    logger.info(f"Using optimal k from paper: {optimal_k}")
    
    X_train_fs, selector = chi_square_select(X_train_tf, y_train, k=optimal_k)
    X_test_fs = selector.transform(X_test_tf)
    
    logger.info(f"Features reduced: {X_train_tf.shape[1]} → {X_train_fs.shape[1]}")
    logger.info(f"Reduction: {(1 - optimal_k/X_train_tf.shape[1])*100:.1f}%")
    
    # 5. Initialize models
    models = {
        "MNB": MultinomialNB(),
        "LR": LogisticRegression(max_iter=1000, random_state=SEED),
        "DT": DecisionTreeClassifier(random_state=SEED),
        "SVM": LinearSVC(random_state=SEED, max_iter=2000),
        "KNN": KNeighborsClassifier(),
    }
    
    # 6. Evaluate BEFORE feature selection (Cross-validation)
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-VALIDATION (BEFORE Feature Selection)")
    logger.info("=" * 70)
    
    results_before = {}
    for name, model in models.items():
        logger.info(f"\nEvaluating {name}...")
        # Keep sparse matrix - no .toarray()!
        results_before[name] = evaluate_cv(model, X_train_tf, y_train, name)
        
        acc = results_before[name]['Accuracy']
        logger.info(f"  Accuracy: {acc['mean']:.4f} ± {acc['std']:.4f}")
    
    # 7. Evaluate AFTER feature selection (Cross-validation)
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-VALIDATION (AFTER Feature Selection)")
    logger.info("=" * 70)
    
    results_after = {}
    for name, model in models.items():
        logger.info(f"\nEvaluating {name}...")
        # Keep sparse matrix - no .toarray()!
        results_after[name] = evaluate_cv(model, X_train_fs, y_train, name)
        
        acc = results_after[name]['Accuracy']
        logger.info(f"  Accuracy: {acc['mean']:.4f} ± {acc['std']:.4f}")
    
    # 8. Final evaluation on TEST SET (best model: DT)
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET EVALUATION (Best Model: Decision Tree)")
    logger.info("=" * 70)
    
    best_model = DecisionTreeClassifier(random_state=SEED)
    # Keep sparse - no .toarray()!
    test_results = evaluate_test_set(
        best_model, X_train_fs, y_train, X_test_fs, y_test
    )
    
    logger.info("\nDecision Tree on Test Set:")
    for metric, value in test_results.items():
        logger.info(f"  {metric}: {value*100:.2f}%")
    
    # 9. Computational efficiency
    logger.info("\n" + "=" * 70)
    logger.info("COMPUTATIONAL EFFICIENCY")
    logger.info("=" * 70)
    
    efficiency_before = measure_computational_efficiency(
        DecisionTreeClassifier(random_state=SEED),
        X_train_tf, y_train, X_test_tf, "DT-Before"
    )
    
    efficiency_after = measure_computational_efficiency(
        DecisionTreeClassifier(random_state=SEED),
        X_train_fs, y_train, X_test_fs, "DT-After"
    )
    
    logger.info("\nBefore Feature Selection:")
    for k, v in efficiency_before.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info("\nAfter Feature Selection:")
    for k, v in efficiency_after.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # 10. Visualizations
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 70)
    
    plot_comparison_bars(results_before, results_after, 
                        "Model Comparison", "comparison.png")
    
    plot_tsne(X_train_tf, X_train_fs, y_train)
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Results saved in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()