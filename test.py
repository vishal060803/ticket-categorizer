import csv
import tempfile
import unittest
from pathlib import Path

from main import (
    ALLOWED_LABELS,
    FINAL_OUTPUT_FIELDS,
    REQUIRED_COLUMNS,
    ValidationError,
    build_tfidf_vectorizer,
    choose_best_model,
    clean_text,
    extract_confusion_pairs,
    get_phase6_sample_tickets,
    get_priority_tag,
    lock_phase1_objectives,
    predict_sample_tickets,
    predict_ticket,
    preprocess_rows,
    train_phase3_models,
    validate_columns,
    validate_dataset,
    validate_labels,
)

def _has_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


class TestSetupAndPreparation(unittest.TestCase):
    def test_phase1_contract_is_defined(self):
        contract = lock_phase1_objectives()
        self.assertEqual(tuple(contract["required_columns"]), tuple(REQUIRED_COLUMNS))
        self.assertEqual(tuple(contract["allowed_labels"]), tuple(ALLOWED_LABELS))
        self.assertEqual(tuple(contract["final_outputs"]), tuple(FINAL_OUTPUT_FIELDS))

    def test_validate_columns_passes(self):
        validate_columns(["subject", "body", "category", "ticket_id"])

    def test_validate_columns_fails(self):
        with self.assertRaises(ValidationError):
            validate_columns(["subject", "body"])

    def test_validate_labels_passes(self):
        rows = [
            {"subject": "invoice issue", "body": "please refund", "category": "Billing"},
            {"subject": "login broken", "body": "cannot access", "category": "Technical"},
            {"subject": "leave policy", "body": "need clarification", "category": "HR"},
            {"subject": "hello", "body": "general question", "category": "General"},
        ]
        validate_labels(rows)

    def test_validate_labels_fails(self):
        rows = [{"subject": "test", "body": "test", "category": "Unknown"}]
        with self.assertRaises(ValidationError):
            validate_labels(rows)

    def test_clean_text_applies_phase2_rules(self):
        value = clean_text("PLEASE!! Refund me for INVOICE #123, ASAP!!!")
        self.assertEqual(value, "refund invoice 123 asap")

    def test_preprocess_rows_merges_and_cleans(self):
        rows = [
            {
                "subject": "Payment failed",
                "body": "My card was charged twice.",
                "category": "Billing",
            }
        ]

        processed = preprocess_rows(rows)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]["text_raw"], "Payment failed My card was charged twice.")
        self.assertEqual(processed[0]["text_clean"], "payment failed card charged twice")

    def test_preprocess_rows_fails_on_empty_subject_and_body(self):
        rows = [{"subject": "   ", "body": "", "category": "General"}]
        with self.assertRaises(ValidationError):
            preprocess_rows(rows)

    def test_preprocess_rows_fails_when_cleaned_text_is_empty(self):
        rows = [{"subject": "the and is", "body": "please the", "category": "General"}]
        with self.assertRaises(ValidationError):
            preprocess_rows(rows)

    def test_validate_dataset_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "tickets.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=["subject", "body", "category"])
                writer.writeheader()
                writer.writerow(
                    {
                        "subject": "Payment failed",
                        "body": "Card charged twice",
                        "category": "Billing",
                    }
                )
                writer.writerow(
                    {
                        "subject": "Laptop not booting",
                        "body": "Screen stays black",
                        "category": "Technical",
                    }
                )

            result = validate_dataset(csv_path)
            self.assertIn("final_outputs", result)


@unittest.skipUnless(_has_sklearn(), "scikit-learn not installed")
class TestPhase3To6ModelBuild(unittest.TestCase):
    def _sample_rows(self):
        return [
            {"subject": "Invoice charged twice", "body": "Need refund now", "category": "Billing"},
            {"subject": "Billing discrepancy", "body": "wrong amount on invoice", "category": "Billing"},
            {"subject": "Server down", "body": "application not working", "category": "Technical"},
            {"subject": "Cannot login", "body": "password reset link fails", "category": "Technical"},
            {"subject": "Interview schedule", "body": "share available slots", "category": "HR"},
            {"subject": "Payroll issue", "body": "salary slip missing", "category": "HR"},
            {"subject": "Need information", "body": "general account question", "category": "General"},
            {"subject": "Policy clarification", "body": "where can I find handbook", "category": "General"},
        ]

    def test_build_tfidf_vectorizer_accepts_bigrams(self):
        vec = build_tfidf_vectorizer(use_bigrams=True)
        self.assertEqual(vec.ngram_range, (1, 2))

    def test_choose_best_model_not_accuracy_only(self):
        evaluations = {
            "ModelA": {"f1_weighted": 0.70, "recall_macro": 0.80, "accuracy": 0.95},
            "ModelB": {"f1_weighted": 0.82, "recall_macro": 0.75, "accuracy": 0.90},
        }
        best = choose_best_model(evaluations)
        self.assertEqual(best, "ModelB")

    def test_extract_confusion_pairs_returns_top_pairs(self):
        matrix = [
            [3, 2, 0, 0],
            [1, 4, 1, 0],
            [0, 2, 5, 0],
            [0, 0, 1, 6],
        ]
        pairs = extract_confusion_pairs(matrix, ALLOWED_LABELS, top_n=2)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["count"], 2)
        self.assertEqual(pairs[0]["actual"], "Billing")
        self.assertEqual(pairs[0]["predicted"], "Technical")

    def test_train_phase3_models_returns_evaluation_details(self):
        processed = preprocess_rows(self._sample_rows())
        result = train_phase3_models(processed_rows=processed, use_bigrams=False)

        self.assertIn("MultinomialNB", result["evaluations"])
        self.assertIn("LogisticRegression", result["evaluations"])
        self.assertIn(result["best_model_name"], ("MultinomialNB", "LogisticRegression"))

        for model_name in ("MultinomialNB", "LogisticRegression"):
            metrics = result["evaluations"][model_name]
            self.assertIn("accuracy", metrics)
            self.assertIn("precision_macro", metrics)
            self.assertIn("recall_macro", metrics)
            self.assertIn("f1_macro", metrics)
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("common_misclassifications", metrics)
            self.assertIn("risky_confusion_pairs", metrics)
            self.assertIn("per_class", metrics)

    def test_get_priority_tag_urgent_and_normal(self):
        self.assertEqual(get_priority_tag("server down outage"), "urgent")
        self.assertEqual(get_priority_tag("need clarification handbook policy"), "normal")

    def test_predict_ticket_output_schema_and_rules(self):
        processed = preprocess_rows(self._sample_rows())
        trained = train_phase3_models(processed_rows=processed, use_bigrams=False)

        result = predict_ticket(
            trained_bundle=trained,
            subject="Urgent: production server down",
            body="App is not working and users cannot access portal",
            review_threshold=0.60,
        )

        for required_key in FINAL_OUTPUT_FIELDS:
            self.assertIn(required_key, result)

        self.assertIn(result["predicted_category"], ALLOWED_LABELS)
        self.assertGreaterEqual(result["confidence_percent"], 0.0)
        self.assertLessEqual(result["confidence_percent"], 100.0)
        self.assertEqual(result["priority_tag"], "urgent")

        expected_review = (result["confidence_percent"] / 100) < 0.60
        self.assertEqual(result["needs_human_review"], expected_review)

    def test_get_phase6_sample_tickets_has_required_shape(self):
        tickets = get_phase6_sample_tickets()
        self.assertGreaterEqual(len(tickets), 5)
        self.assertTrue(all("subject" in item and "body" in item for item in tickets))

        ambiguous_present = any(
            "unsure" in item["body"].lower() or "not sure" in item["body"].lower()
            for item in tickets
        )
        self.assertTrue(ambiguous_present)

    def test_predict_sample_tickets_returns_predictions_for_each_sample(self):
        processed = preprocess_rows(self._sample_rows())
        trained = train_phase3_models(processed_rows=processed, use_bigrams=False)

        sample_predictions = predict_sample_tickets(
            trained_bundle=trained,
            review_threshold=0.60,
        )

        self.assertGreaterEqual(len(sample_predictions), 5)
        for index, item in enumerate(sample_predictions, start=1):
            self.assertEqual(item["sample_id"], index)
            for required_key in FINAL_OUTPUT_FIELDS:
                self.assertIn(required_key, item)
            self.assertIn(item["predicted_category"], ALLOWED_LABELS)
            self.assertGreaterEqual(item["confidence_percent"], 0.0)
            self.assertLessEqual(item["confidence_percent"], 100.0)


    def test_predict_ticket_invalid_threshold_and_empty_text(self):
        processed = preprocess_rows(self._sample_rows())
        trained = train_phase3_models(processed_rows=processed, use_bigrams=False)

        with self.assertRaises(ValidationError):
            predict_ticket(
                trained_bundle=trained,
                subject="hello",
                body="world",
                review_threshold=1.2,
            )

        with self.assertRaises(ValidationError):
            predict_ticket(
                trained_bundle=trained,
                subject="the and is",
                body="please the",
                review_threshold=0.6,
            )


if __name__ == "__main__":
    unittest.main()





