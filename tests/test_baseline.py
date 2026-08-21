from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from support_sense.baseline import build_baseline_pipeline
from support_sense.metrics import calculate_classification_metrics


def test_baseline_pipeline_contains_expected_steps():
    """The baseline must use TF-IDF followed by Logistic Regression."""

    pipeline = build_baseline_pipeline()

    # Verify the feature extraction stage.
    assert isinstance(
        pipeline.named_steps["tfidf"],
        TfidfVectorizer,
    )

    # Verify the classifier stage.
    assert isinstance(
        pipeline.named_steps["classifier"],
        LogisticRegression,
    )


def test_baseline_uses_unigrams():
    """The first baseline should use unigram features only."""

    pipeline = build_baseline_pipeline()

    vectorizer = pipeline.named_steps["tfidf"]

    assert vectorizer.ngram_range == (1, 1)


def test_baseline_does_not_remove_stop_words():
    """Stop-word removal is deliberately excluded from baseline 001."""

    pipeline = build_baseline_pipeline()

    vectorizer = pipeline.named_steps["tfidf"]

    assert vectorizer.stop_words is None


def test_baseline_can_fit_and_predict():
    """The complete text pipeline should support fit and prediction."""

    texts = [
        "payment card invoice",
        "charged payment billing",
        "package delivery tracking",
        "courier shipment package",
        "password login account",
        "account password reset",
    ]

    labels = [
        "billing",
        "billing",
        "delivery",
        "delivery",
        "account_access",
        "account_access",
    ]

    pipeline = build_baseline_pipeline()

    # Fitting should train both TF-IDF and the classifier.
    pipeline.fit(texts, labels)

    # Prediction should accept raw text rather than manually vectorized data.
    predictions = pipeline.predict(["my package tracking is delayed"])

    assert len(predictions) == 1


def test_perfect_predictions_produce_perfect_metrics():
    """Perfect predictions should produce metric values of 1.0."""

    y_true = [
        "billing",
        "delivery",
        "account_access",
    ]

    y_pred = [
        "billing",
        "delivery",
        "account_access",
    ]

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0
