import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import (
    ComplementNB,
    MultinomialNB,
)
from sklearn.svm import LinearSVC

from support_sense.experiments import run_experiment
from support_sense.modeling import (
    build_text_pipeline,
    create_classifier,
)


def test_pipeline_accepts_bigram_configuration():
    """The pipeline should preserve requested TF-IDF settings."""

    pipeline = build_text_pipeline(
        LogisticRegression(),
        ngram_range=(1, 2),
    )

    vectorizer = pipeline.named_steps["tfidf"]

    # Verify that experiment configuration reached the vectorizer.
    assert vectorizer.ngram_range == (1, 2)


def test_pipeline_accepts_document_frequency_limits():
    """min_df and max_df should be configurable for experiments."""

    pipeline = build_text_pipeline(
        LogisticRegression(),
        min_df=2,
        max_df=0.90,
    )

    vectorizer = pipeline.named_steps["tfidf"]

    assert vectorizer.min_df == 2
    assert vectorizer.max_df == 0.90


def test_pipeline_accepts_sublinear_tf():
    """Sublinear TF should be available as an experiment option."""

    pipeline = build_text_pipeline(
        LogisticRegression(),
        sublinear_tf=True,
    )

    vectorizer = pipeline.named_steps["tfidf"]

    assert vectorizer.sublinear_tf is True


def test_classifier_factory_builds_supported_models():
    """Every declared experiment classifier should be constructible."""

    assert isinstance(
        create_classifier("logistic_regression"),
        LogisticRegression,
    )

    assert isinstance(
        create_classifier("multinomial_nb"),
        MultinomialNB,
    )

    assert isinstance(
        create_classifier("complement_nb"),
        ComplementNB,
    )

    assert isinstance(
        create_classifier("linear_svc"),
        LinearSVC,
    )


def test_classifier_factory_rejects_unknown_classifier():
    """Typos should fail rather than silently changing experiments."""

    with pytest.raises(ValueError):
        create_classifier("magic_classifier")


def test_experiment_runner_returns_metrics():
    """One experiment should train and return evaluation metadata."""

    pipeline = build_text_pipeline(create_classifier("logistic_regression"))

    X_train = [
        "payment invoice card",
        "billing payment charge",
        "package courier delivery",
        "shipment package tracking",
    ]

    y_train = [
        "billing",
        "billing",
        "delivery",
        "delivery",
    ]

    X_validation = [
        "payment card",
        "package tracking",
    ]

    y_validation = [
        "billing",
        "delivery",
    ]

    result, fitted_model = run_experiment(
        experiment_id="TEST-001",
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    # Ensure essential experiment metadata is returned.
    assert result["experiment_id"] == "TEST-001"
    assert "f1_macro" in result
    assert "training_seconds" in result
    assert "vocabulary_size" in result

    # Ensure the returned pipeline has actually been fitted.
    assert hasattr(
        fitted_model.named_steps["classifier"],
        "classes_",
    )


def test_priority_model_predicts_known_labels():
    """Priority predictions must belong to the supported class set."""

    texts = [
        "general product question",
        "simple account question",
        "need help today immediately",
        "urgent payment failure",
        "delivery information request",
        "critical delivery problem",
    ]

    priorities = [
        "low",
        "low",
        "medium",
        "high",
        "medium",
        "high",
    ]

    model = build_text_pipeline(create_classifier("logistic_regression"))

    model.fit(
        texts,
        priorities,
    )

    predictions = model.predict(["urgent account problem"])

    assert predictions[0] in {
        "low",
        "medium",
        "high",
    }
