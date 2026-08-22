import pandas as pd
import pytest

from support_sense.baseline import build_baseline_pipeline
from support_sense.evaluation import build_error_analysis


@pytest.fixture
def fitted_model():
    """Return a small fitted model for evaluation tests."""
    training_texts = [
        "payment invoice charge",
        "billing card payment",
        "package tracking courier",
        "delivery shipment package",
    ]
    training_labels = [
        "billing",
        "billing",
        "delivery",
        "delivery",
    ]

    model = build_baseline_pipeline()
    model.fit(training_texts, training_labels)

    return model


@pytest.fixture
def validation_df():
    """Return a small validation dataset."""
    return pd.DataFrame(
        {
            "ticket_id": ["TKT-TEST-1"],
            "text": ["package payment issue"],
            "category": ["billing"],
        }
    )


@pytest.fixture
def error_analysis(validation_df, fitted_model):
    """Return error-analysis results for the fixture data."""
    return build_error_analysis(
        validation_df,
        fitted_model,
    )


def test_error_analysis_has_expected_columns(error_analysis):
    """Error-analysis output should expose required diagnostic fields."""
    expected_columns = {
        "ticket_id",
        "text",
        "actual_category",
        "predicted_category",
        "confidence",
    }

    assert expected_columns.issubset(error_analysis.columns)


def test_error_confidence_is_probability(error_analysis):
    """Reported model probability should remain between zero and one."""
    if not error_analysis.empty:
        assert error_analysis["confidence"].between(0.0, 1.0).all()


def test_error_analysis_contains_only_wrong_predictions(error_analysis):
    """Correct classifications must not appear in the error report."""

    assert (
        error_analysis["actual_category"] != error_analysis["predicted_category"]
    ).all()
