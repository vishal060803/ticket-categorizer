from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REQUIRED_COLUMNS: Sequence[str] = ("subject", "body", "category")
ALLOWED_LABELS: Sequence[str] = ("Billing", "Technical", "HR", "General")
FINAL_OUTPUT_FIELDS: Sequence[str] = (
    "predicted_category",
    "confidence_percent",
    "needs_human_review",
    "priority_tag",
)

PLANNED_LIBRARIES: Sequence[str] = (
    "pandas",
    "scikit-learn",
    "numpy",
    "streamlit (optional)",
)

PHASE3_REQUIRED_PACKAGES: Sequence[str] = (
    "scikit-learn",
    "numpy",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "with",
    "you",
    "your",
}


URGENT_KEYWORDS: Sequence[str] = (
    "down",
    "urgent",
    "asap",
    "outage",
    "cannot access",
    "not working",
)
DEFAULT_REVIEW_THRESHOLD = 0.60

class ValidationError(ValueError):
    """Raised when dataset setup validation fails."""


def read_csv_rows(csv_path: str | Path) -> List[Dict[str, str]]:
    """Read a CSV file and return rows as dictionaries."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValidationError("CSV has no header row.")
        return list(reader)


def validate_columns(fieldnames: Iterable[str]) -> None:
    """Validate required columns exist in CSV header."""
    found = set(fieldnames)
    missing = [column for column in REQUIRED_COLUMNS if column not in found]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")


def validate_labels(rows: Iterable[Dict[str, str]]) -> None:
    """Validate all label values are inside the 4 allowed categories."""
    allowed = set(ALLOWED_LABELS)
    invalid_labels = sorted(
        {
            (row.get("category") or "").strip()
            for row in rows
            if (row.get("category") or "").strip() not in allowed
        }
    )

    if invalid_labels:
        raise ValidationError(
            f"Found invalid categories: {invalid_labels}. Allowed: {list(ALLOWED_LABELS)}"
        )


def merge_subject_body(row: Dict[str, str]) -> str:
    """Merge ticket subject and body into one raw text field."""
    subject = (row.get("subject") or "").strip()
    body = (row.get("body") or "").strip()
    return f"{subject} {body}".strip()


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/noise, normalize spaces, remove stopwords."""
    lowered = text.lower()
    letters_and_digits = re.sub(r"[^a-z0-9\s]", " ", lowered)
    collapsed_spaces = re.sub(r"\s+", " ", letters_and_digits).strip()

    if not collapsed_spaces:
        return ""

    tokens = [token for token in collapsed_spaces.split(" ") if token and token not in STOPWORDS]
    return " ".join(tokens)


def get_priority_tag(text_clean: str) -> str:
    """Return urgent/normal priority tag based on keyword rules."""
    if not text_clean:
        return "normal"

    tokens = set(text_clean.split())
    for keyword in URGENT_KEYWORDS:
        if " " in keyword:
            if keyword in text_clean:
                return "urgent"
        elif keyword in tokens:
            return "urgent"

    return "normal"


def sanity_check_rows(rows: Iterable[Dict[str, str]]) -> None:
    """Quick checks for null/empty ticket text and category consistency."""
    row_list = list(rows)
    validate_labels(row_list)

    for index, row in enumerate(row_list, start=1):
        merged_raw = merge_subject_body(row)
        if not merged_raw:
            raise ValidationError(
                f"Row {index} has empty subject and body. Provide ticket text before training."
            )


def preprocess_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Prepare dataset rows for modeling in later phases."""
    row_list = list(rows)
    sanity_check_rows(row_list)

    processed_rows: List[Dict[str, str]] = []
    for index, row in enumerate(row_list, start=1):
        category = (row.get("category") or "").strip()
        text_raw = merge_subject_body(row)
        text_clean = clean_text(text_raw)

        if not text_clean:
            raise ValidationError(
                f"Row {index} became empty after cleaning. Add more descriptive text."
            )

        processed_rows.append(
            {
                "subject": (row.get("subject") or "").strip(),
                "body": (row.get("body") or "").strip(),
                "category": category,
                "text_raw": text_raw,
                "text_clean": text_clean,
            }
        )

    return processed_rows


def _import_sklearn() -> Dict[str, Any]:
    """Import sklearn modules lazily so Phase 1/2 can run without sklearn."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
    except ImportError as error:
        raise RuntimeError(
            "Phase 3/4 requires scikit-learn and numpy. Install with: pip install -r requirements.txt"
        ) from error

    return {
        "TfidfVectorizer": TfidfVectorizer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "train_test_split": train_test_split,
        "MultinomialNB": MultinomialNB,
    }


def build_tfidf_vectorizer(use_bigrams: bool = False) -> Any:
    """Create TF-IDF vectorizer (unigrams; optional bigrams)."""
    modules = _import_sklearn()
    TfidfVectorizer = modules["TfidfVectorizer"]
    ngram_range = (1, 2) if use_bigrams else (1, 1)
    return TfidfVectorizer(ngram_range=ngram_range)


def split_text_and_labels(
    processed_rows: Sequence[Dict[str, str]],
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split cleaned text and labels for train/test, stratifying when feasible."""
    modules = _import_sklearn()
    train_test_split = modules["train_test_split"]

    texts = [row["text_clean"] for row in processed_rows]
    labels = [row["category"] for row in processed_rows]

    label_counts = Counter(labels)
    class_count = len(set(labels))
    sample_count = len(labels)

    if isinstance(test_size, float):
        test_count = math.ceil(sample_count * test_size)
    else:
        test_count = int(test_size)

    train_count = sample_count - test_count
    has_min_samples_per_class = all(count >= 2 for count in label_counts.values())
    split_can_cover_all_classes = test_count >= class_count and train_count >= class_count
    can_stratify = class_count > 1 and has_min_samples_per_class and split_can_cover_all_classes
    stratify_target = labels if can_stratify else None

    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def extract_confusion_pairs(
    confusion_matrix_values: Sequence[Sequence[int]],
    label_order: Sequence[str],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Return top off-diagonal confusion pairs sorted by frequency."""
    pairs: List[Dict[str, Any]] = []

    for row_index, actual_label in enumerate(label_order):
        for column_index, predicted_label in enumerate(label_order):
            if row_index == column_index:
                continue
            count = int(confusion_matrix_values[row_index][column_index])
            if count > 0:
                pairs.append(
                    {
                        "actual": actual_label,
                        "predicted": predicted_label,
                        "count": count,
                    }
                )

    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:top_n]


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_order: Sequence[str],
) -> Dict[str, Any]:
    """Compute evaluation metrics for model comparison and reporting."""
    modules = _import_sklearn()
    accuracy_score = modules["accuracy_score"]
    confusion_matrix = modules["confusion_matrix"]
    precision_recall_fscore_support = modules["precision_recall_fscore_support"]

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(label_order),
            average=None,
            zero_division=0,
        )
    )

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for index, label in enumerate(label_order):
        per_class_metrics[label] = {
            "precision": float(per_class_precision[index]),
            "recall": float(per_class_recall[index]),
            "f1": float(per_class_f1[index]),
            "support": float(per_class_support[index]),
        }

    matrix = confusion_matrix(y_true, y_pred, labels=list(label_order)).tolist()
    common_pairs = extract_confusion_pairs(matrix, label_order=label_order, top_n=5)

    risky_pairs = [
        pair
        for pair in common_pairs
        if pair["actual"] in {"Technical", "Billing"} or pair["predicted"] in {"Technical", "Billing"}
    ]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class_metrics,
        "confusion_matrix": matrix,
        "common_misclassifications": common_pairs,
        "risky_confusion_pairs": risky_pairs,
    }


def choose_best_model(evaluations: Dict[str, Dict[str, Any]]) -> str:
    """Pick best model by weighted F1, then macro recall, then accuracy."""
    if not evaluations:
        raise ValidationError("No model evaluations found to choose best model.")

    ranked = sorted(
        evaluations.items(),
        key=lambda item: (
            item[1]["f1_weighted"],
            item[1]["recall_macro"],
            item[1]["accuracy"],
        ),
        reverse=True,
    )
    return ranked[0][0]


def train_phase3_models(
    processed_rows: Sequence[Dict[str, str]],
    use_bigrams: bool = False,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and compare MultinomialNB vs LogisticRegression on TF-IDF features."""
    if len(processed_rows) < 4:
        raise ValidationError("Need at least 4 cleaned rows to train and compare models.")

    modules = _import_sklearn()
    MultinomialNB = modules["MultinomialNB"]
    LogisticRegression = modules["LogisticRegression"]

    x_train, x_test, y_train, y_test = split_text_and_labels(
        processed_rows=processed_rows,
        test_size=test_size,
        random_state=random_state,
    )

    vectorizer = build_tfidf_vectorizer(use_bigrams=use_bigrams)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
    }

    evaluations: Dict[str, Dict[str, Any]] = {}
    trained_models: Dict[str, Any] = {}

    for model_name, model in models.items():
        model.fit(x_train_vec, y_train)
        predictions = model.predict(x_test_vec)
        evaluations[model_name] = evaluate_predictions(y_test, predictions, label_order=ALLOWED_LABELS)
        trained_models[model_name] = model

    best_model_name = choose_best_model(evaluations)

    return {
        "vectorizer": vectorizer,
        "models": trained_models,
        "evaluations": evaluations,
        "best_model_name": best_model_name,
        "best_model": trained_models[best_model_name],
        "x_test": x_test,
        "y_test": y_test,
    }


def predict_ticket(
    trained_bundle: Dict[str, Any],
    subject: str,
    body: str,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> Dict[str, Any]:
    """Predict category with confidence, human-review flag, and priority tag."""
    if review_threshold < 0 or review_threshold > 1:
        raise ValidationError("review_threshold must be between 0 and 1.")

    text_raw = merge_subject_body({"subject": subject, "body": body})
    text_clean = clean_text(text_raw)
    if not text_clean:
        raise ValidationError("Incoming ticket text is empty after cleaning.")

    vectorizer = trained_bundle["vectorizer"]
    model = trained_bundle["best_model"]
    vector = vectorizer.transform([text_clean])

    predicted_category = str(model.predict(vector)[0])
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        classes = list(model.classes_)
        predicted_index = classes.index(predicted_category)
        confidence = float(probabilities[predicted_index])

    result = {
        "predicted_category": predicted_category,
        "confidence_percent": round(confidence * 100, 2),
        "needs_human_review": confidence < review_threshold,
        "priority_tag": get_priority_tag(text_clean),
        "text_clean": text_clean,
    }
    return result




def get_phase6_sample_tickets() -> List[Dict[str, str]]:
    """Return unseen sample tickets (mixed categories + one ambiguous case)."""
    return [
        {
            "subject": "Refund request for duplicate charge",
            "body": "I was billed twice for last month, please reverse one charge.",
        },
        {
            "subject": "VPN not working for remote access",
            "body": "Cannot access internal tools since morning, urgent help needed.",
        },
        {
            "subject": "Question about leave balance",
            "body": "Could you confirm my remaining paid leave days for this quarter?",
        },
        {
            "subject": "Need account handbook link",
            "body": "I am looking for the policy document location and guidance.",
        },
        {
            "subject": "Portal issue and payroll confusion",
            "body": "I am unsure if this is HR or technical, the portal is slow and my salary slip is missing.",
        },
        {
            "subject": "Invoice mismatch for annual subscription",
            "body": "The renewal amount does not match the quoted plan price.",
        },
    ]


def predict_sample_tickets(
    trained_bundle: Dict[str, Any],
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    sample_tickets: Sequence[Dict[str, str]] | None = None,
) -> List[Dict[str, Any]]:
    """Predict category and routing signals for sample tickets."""
    tickets = list(sample_tickets) if sample_tickets is not None else get_phase6_sample_tickets()
    predictions: List[Dict[str, Any]] = []

    for index, ticket in enumerate(tickets, start=1):
        result = predict_ticket(
            trained_bundle=trained_bundle,
            subject=ticket.get("subject", ""),
            body=ticket.get("body", ""),
            review_threshold=review_threshold,
        )
        predictions.append(
            {
                "sample_id": index,
                "subject": ticket.get("subject", ""),
                "body": ticket.get("body", ""),
                **result,
            }
        )

    return predictions


def _print_sample_predictions(sample_predictions: Sequence[Dict[str, Any]]) -> None:
    """Print batch sample predictions in a compact CLI-friendly format."""
    for row in sample_predictions:
        print(
            f"sample_{row['sample_id']}: "
            f"predicted_category={row['predicted_category']}, "
            f"confidence_percent={row['confidence_percent']}, "
            f"needs_human_review={row['needs_human_review']}, "
            f"priority_tag={row['priority_tag']}"
        )


def lock_phase1_objectives() -> Dict[str, Sequence[str]]:
    """Return phase-1 objective contract for the project."""
    return {
        "required_columns": REQUIRED_COLUMNS,
        "allowed_labels": ALLOWED_LABELS,
        "final_outputs": FINAL_OUTPUT_FIELDS,
        "planned_libraries": PLANNED_LIBRARIES,
    }


def validate_dataset(csv_path: str | Path) -> Dict[str, Sequence[str]]:
    """Run full setup validation against dataset file (Phase 1 + 2 checks)."""
    rows = read_csv_rows(csv_path)

    if not rows:
        raise ValidationError("Dataset is empty. Add labeled ticket rows first.")

    validate_columns(rows[0].keys())
    preprocess_rows(rows)
    return lock_phase1_objectives()


def _print_evaluation_summary(model_name: str, metrics: Dict[str, Any]) -> None:
    """Print concise evaluation summary for CLI use."""
    print(f"\n{model_name}")
    print(
        f"accuracy={metrics['accuracy']:.3f}, "
        f"precision_macro={metrics['precision_macro']:.3f}, "
        f"recall_macro={metrics['recall_macro']:.3f}, "
        f"f1_macro={metrics['f1_macro']:.3f}"
    )
    print("confusion_matrix=")
    for row in metrics["confusion_matrix"]:
        print(row)

    if metrics["common_misclassifications"]:
        print("common_misclassifications=")
        for pair in metrics["common_misclassifications"]:
            print(f"{pair['actual']} -> {pair['predicted']} : {pair['count']}")

    if metrics["risky_confusion_pairs"]:
        print("risky_confusion_pairs=")
        for pair in metrics["risky_confusion_pairs"]:
            print(f"{pair['actual']} -> {pair['predicted']} : {pair['count']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1/2/3/4/5/6 setup, model-building, evaluation, and real-time prediction"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to CSV dataset. If omitted, prints objective contract only.",
    )
    parser.add_argument(
        "--train-phase3",
        action="store_true",
        help="Run Phase 3 training (TF-IDF + NB/LR comparison).",
    )
    parser.add_argument(
        "--bigrams",
        action="store_true",
        help="Use TF-IDF bigrams in addition to unigrams.",
    )
    parser.add_argument(
        "--predict-subject",
        type=str,
        default="",
        help="Subject text for one real-time ticket prediction.",
    )
    parser.add_argument(
        "--predict-body",
        type=str,
        default="",
        help="Body text for one real-time ticket prediction.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=DEFAULT_REVIEW_THRESHOLD,
        help="Human-review threshold between 0 and 1 (default: 0.60).",
    )
    parser.add_argument(
        "--run-sample-tickets",
        action="store_true",
        help="Run predictions for default unseen sample tickets (5+ with one ambiguous case).",
    )
    args = parser.parse_args()

    contract = lock_phase1_objectives()

    print("=== Objective Lock ===")
    print(f"Required columns: {list(contract['required_columns'])}")
    print(f"Allowed labels: {list(contract['allowed_labels'])}")
    print(f"Final outputs: {list(contract['final_outputs'])}")
    print(f"Planned libraries: {list(contract['planned_libraries'])}")
    print(f"Phase 3/4/5 packages: {list(PHASE3_REQUIRED_PACKAGES)}")

    if args.data:
        try:
            rows = read_csv_rows(args.data)
            validate_columns(rows[0].keys() if rows else [])
            processed = preprocess_rows(rows)
            print(f"\nDataset validation passed: {args.data}")
            print(f"Prepared rows: {len(processed)}")
            print(f"Sample cleaned text: {processed[0]['text_clean']}")

            if args.train_phase3:
                phase3_result = train_phase3_models(processed, use_bigrams=args.bigrams)
                print("\n=== Phase 3/4 Model Comparison ===")
                for name, metrics in phase3_result["evaluations"].items():
                    _print_evaluation_summary(name, metrics)
                print(f"\nBest model selected: {phase3_result['best_model_name']}")

                if args.predict_subject or args.predict_body:
                    prediction = predict_ticket(
                        trained_bundle=phase3_result,
                        subject=args.predict_subject,
                        body=args.predict_body,
                        review_threshold=args.review_threshold,
                    )
                    print("\n=== Phase 5 Real-time Prediction ===")
                    print(f"predicted_category={prediction['predicted_category']}")
                    print(f"confidence_percent={prediction['confidence_percent']}")
                    print(f"needs_human_review={prediction['needs_human_review']}")
                    print(f"priority_tag={prediction['priority_tag']}")

                if args.run_sample_tickets:
                    sample_predictions = predict_sample_tickets(
                        trained_bundle=phase3_result,
                        review_threshold=args.review_threshold,
                    )
                    print("\n=== Final Phase: Sample Ticket Predictions ===")
                    _print_sample_predictions(sample_predictions)
            elif args.predict_subject or args.predict_body or args.run_sample_tickets:
                raise ValidationError(
                    "Use --train-phase3 when requesting prediction flags."
                )
        except (ValidationError, FileNotFoundError, RuntimeError) as error:
            print(f"\nExecution failed: {error}")
            raise SystemExit(1) from error
if __name__ == "__main__":
    main()
















