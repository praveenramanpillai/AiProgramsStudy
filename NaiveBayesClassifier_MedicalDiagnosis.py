"""
Medical Diagnosis Naive Bayes (Categorical)

Features (examples):
  Fever: Yes/No
  Cough: Yes/No
  SoreThroat: Yes/No
  BodyAche: Yes/No
  Fatigue: Yes/No

Class label:
  Diagnosis: Flu / Cold / Allergy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from collections import Counter, defaultdict
import math

# --- scikit-learn comparison ---
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Sample Dataset
# -----------------------------
FEATURES = ["Fever", "Cough", "SoreThroat", "BodyAche", "Fatigue"]
LABEL = "Diagnosis"

DATASET: List[Dict[str, str]] = [
    # Flu Symptoms
    {"Fever": "Yes", "Cough": "Yes", "SoreThroat": "No",  "BodyAche": "Yes", "Fatigue": "Yes", "Diagnosis": "Flu"},
    {"Fever": "Yes", "Cough": "Yes", "SoreThroat": "Yes", "BodyAche": "Yes", "Fatigue": "Yes", "Diagnosis": "Flu"},
    {"Fever": "Yes", "Cough": "No",  "SoreThroat": "Yes", "BodyAche": "Yes", "Fatigue": "Yes", "Diagnosis": "Flu"},
    {"Fever": "Yes", "Cough": "Yes", "SoreThroat": "No",  "BodyAche": "Yes", "Fatigue": "No",  "Diagnosis": "Flu"},
    {"Fever": "Yes", "Cough": "Yes", "SoreThroat": "Yes", "BodyAche": "No",  "Fatigue": "Yes", "Diagnosis": "Flu"},

    # Cold Symptoms
    {"Fever": "No",  "Cough": "Yes", "SoreThroat": "Yes", "BodyAche": "No",  "Fatigue": "No",  "Diagnosis": "Cold"},
    {"Fever": "No",  "Cough": "Yes", "SoreThroat": "No",  "BodyAche": "No",  "Fatigue": "Yes", "Diagnosis": "Cold"},
    {"Fever": "Yes", "Cough": "Yes", "SoreThroat": "Yes", "BodyAche": "No",  "Fatigue": "No",  "Diagnosis": "Cold"},
    {"Fever": "No",  "Cough": "No",  "SoreThroat": "Yes", "BodyAche": "No",  "Fatigue": "No",  "Diagnosis": "Cold"},
    {"Fever": "No",  "Cough": "Yes", "SoreThroat": "Yes", "BodyAche": "Yes", "Fatigue": "No",  "Diagnosis": "Cold"},

    # Allergy Symptoms
    {"Fever": "No",  "Cough": "No",  "SoreThroat": "Yes", "BodyAche": "No",  "Fatigue": "No",  "Diagnosis": "Allergy"},
    {"Fever": "No",  "Cough": "Yes", "SoreThroat": "No",  "BodyAche": "No",  "Fatigue": "No",  "Diagnosis": "Allergy"},
    {"Fever": "No",  "Cough": "No",  "SoreThroat": "No",  "BodyAche": "No",  "Fatigue": "Yes", "Diagnosis": "Allergy"},
    {"Fever": "No",  "Cough": "Yes", "SoreThroat": "Yes", "BodyAche": "No",  "Fatigue": "Yes", "Diagnosis": "Allergy"},
    {"Fever": "No",  "Cough": "No",  "SoreThroat": "Yes", "BodyAche": "Yes", "Fatigue": "No",  "Diagnosis": "Allergy"},
]


@dataclass
class FrequencyTables:
    class_counts: Counter
    feature_value_counts: Dict[str, Dict[str, Counter]] # Counts how many samples belong to each diagnosis


def build_frequency_tables(dataset: List[Dict[str, str]], features: List[str], label: str) -> FrequencyTables:
    class_counts = Counter()
    feature_value_counts: Dict[str, Dict[str, Counter]] = {f: defaultdict(Counter) for f in features} # Stores counts of each diagnosis for every patient

    for row in dataset:
        c = row[label]
        class_counts[c] += 1
        for f in features:
            v = row[f]
            feature_value_counts[f][c][v] += 1

    return FrequencyTables(class_counts, feature_value_counts)

# Determines all possible values (Yes, No) for each feature
def unique_values_by_feature(dataset: List[Dict[str, str]], features: List[str]) -> Dict[str, List[str]]:
    return {f: sorted({row[f] for row in dataset}) for f in features}

# computes P(Class)= total samples/count of class
def compute_priors(class_counts: Counter) -> Dict[str, float]:
    total = sum(class_counts.values())
    return {c: class_counts[c] / total for c in class_counts}


# Likelihood Table with Laplace Smoothing
def compute_likelihoods(
    freq: FrequencyTables,
    feature_values: Dict[str, List[str]],
    alpha: float = 1.0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Likelihoods with Laplace smoothing:
      P(f=v | class=c) = (count(f=v,c) + alpha) / (count(c) + alpha*K) where K = number of unique values for feature f
    """
    likelihoods: Dict[str, Dict[str, Dict[str, float]]] = {}

    for f in freq.feature_value_counts:
        K = len(feature_values[f])
        likelihoods[f] = {}
        for c in freq.class_counts:
            denom = freq.class_counts[c] + alpha * K
            likelihoods[f][c] = {}
            for v in feature_values[f]:
                num = freq.feature_value_counts[f][c][v] + alpha
                likelihoods[f][c][v] = num / denom

    return likelihoods


# Posterior Probability Calculation
def posterior(
    x: Dict[str, str],
    priors: Dict[str, float],
    likelihoods: Dict[str, Dict[str, Dict[str, float]]],
    features: List[str],
) -> Dict[str, float]:
    """
    Posterior (normalized): P(c|x) ∝ P(c) * Π P(f=v | c)
    Computed in log-space for numerical stability.
    """
    log_scores: Dict[str, float] = {}

    for c, prior in priors.items():
        s = math.log(prior)
        for f in features:
            v = x[f]
            s += math.log(likelihoods[f][c][v])
        log_scores[c] = s

    # Normalize with log-sum-exp
    m = max(log_scores.values())
    exp_scores = {c: math.exp(log_scores[c] - m) for c in log_scores}
    total = sum(exp_scores.values())
    return {c: exp_scores[c] / total for c in exp_scores}


# Printing Helpers
def print_frequency_tables(freq: FrequencyTables, features: List[str]) -> None:
    print("\n=== Frequency Table: Class Counts ===")
    for c, n in freq.class_counts.items():
        print(f"  {c}: {n}")

    print("\n=== Frequency Tables: Feature Value Counts by Class ===")
    for f in features:
        print(f"\nFeature: {f}")
        for c in freq.class_counts:
            print(f"  Class={c}: {dict(freq.feature_value_counts[f][c])}")


def print_likelihood_slice(
    likelihoods: Dict[str, Dict[str, Dict[str, float]]],
    feature_values: Dict[str, List[str]],
    classes: List[str],
    features: List[str],
) -> None:
    print("\n=== Likelihood Tables (with Laplace smoothing) ===")
    for f in features:
        print(f"\nFeature: {f}")
        for c in classes:
            row = {v: round(likelihoods[f][c][v], 4) for v in feature_values[f]}
            print(f"  P({f}=value | {c}): {row}")


# scikit-learn Comparison
def sklearn_demo(dataset: List[Dict[str, str]], features: List[str], label: str) -> None:
    X = [[row[f] for f in features] for row in dataset]
    y = [row[label] for row in dataset]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=21, stratify=y
    )

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_enc = enc.fit_transform(X_train)
    X_test_enc = enc.transform(X_test)

    model = CategoricalNB(alpha=1.0)  # Laplace smoothing
    model.fit(X_train_enc, y_train)

    pred = model.predict(X_test_enc)
    acc = accuracy_score(y_test, pred)

    print("\n=== scikit-learn CategoricalNB Demo ===")
    print(f"Accuracy (holdout test): {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))

    # Show predicted probabilities for one sample
    example = X_test[0]
    proba = model.predict_proba(enc.transform([example]))[0]
    classes = list(model.classes_)
    print("\nExample patient:", dict(zip(features, example)))
    print("Predicted probabilities:", {classes[i]: round(float(proba[i]), 4) for i in range(len(classes))})


# Main Execution
def main() -> None:
    # 1) Frequency tables
    freq = build_frequency_tables(DATASET, FEATURES, LABEL)
    print_frequency_tables(freq, FEATURES)

    # 2) Likelihoods (Laplace correction to prevent zeros)
    feat_vals = unique_values_by_feature(DATASET, FEATURES)
    priors = compute_priors(freq.class_counts)
    likelihoods = compute_likelihoods(freq, feat_vals, alpha=1.0)

    classes_sorted = sorted(freq.class_counts.keys())
    print_likelihood_slice(likelihoods, feat_vals, classes_sorted, FEATURES)

    # 3) Compute posterior probabilities for a patient input
    PATIENT = {"Fever": "Yes",
               "Cough": "Yes",
               "SoreThroat": "No",
               "BodyAche": "Yes",
               "Fatigue": "Yes"}

    post = posterior(PATIENT, priors, likelihoods, FEATURES)

    print("\n=== Posterior Probabilities (manual frequency-table NB) ===")
    print("Patient symptoms:", PATIENT)
    for c in sorted(post.keys()):
        print(f"  P({c} | patient) = {post[c]:.4f}")

    predicted = max(post.items(), key=lambda kv: kv[1])[0]
    print("\nPredicted diagnosis (manual NB):", predicted)

    # 4) scikit-learn comparison
    sklearn_demo(DATASET, FEATURES, LABEL)


if __name__ == "__main__":
    main()
