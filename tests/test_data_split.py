import pandas as pd
import pytest

from support_sense.data_split import split_dataset


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Create a balanced labeled dataset for split tests."""

    rows = []

    # Create enough examples in every class for stratified splitting.
    for category in ["billing", "delivery", "technical_issue"]:
        for number in range(20):
            rows.append(
                {
                    "ticket_id": f"{category}-{number}",
                    "text": f"Example ticket {number}",
                    "category": category,
                }
            )

    return pd.DataFrame(rows)


def test_split_dataset_preserves_all_rows(sample_dataset):
    """Splitting must neither lose nor duplicate records."""

    train_df, validation_df, test_df = split_dataset(sample_dataset)

    total_rows = len(train_df) + len(validation_df) + len(test_df)

    assert total_rows == len(sample_dataset)


def test_split_dataset_has_no_ticket_overlap(sample_dataset):
    """Each ticket must belong to exactly one subset."""

    train_df, validation_df, test_df = split_dataset(sample_dataset)

    train_ids = set(train_df["ticket_id"])
    validation_ids = set(validation_df["ticket_id"])
    test_ids = set(test_df["ticket_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_split_dataset_is_reproducible(sample_dataset):
    """The fixed random state should reproduce the same split."""

    first_train, first_validation, first_test = split_dataset(sample_dataset)

    second_train, second_validation, second_test = split_dataset(sample_dataset)

    assert set(first_train["ticket_id"]) == set(second_train["ticket_id"])

    assert set(first_validation["ticket_id"]) == set(second_validation["ticket_id"])

    assert set(first_test["ticket_id"]) == set(second_test["ticket_id"])


def test_split_dataset_preserves_categories(sample_dataset):
    """Every split should retain every target class."""

    train_df, validation_df, test_df = split_dataset(sample_dataset)

    expected_categories = set(sample_dataset["category"])

    assert set(train_df["category"]) == expected_categories
    assert set(validation_df["category"]) == expected_categories
    assert set(test_df["category"]) == expected_categories
