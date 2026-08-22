import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def build_confusion_table(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    """Create a labeled multiclass confusion matrix."""

    # Rows represent true classes and columns represent predictions.
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return pd.DataFrame(matrix, index=labels, columns=labels)


def extract_confusion_pairs(
    confusion_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return class-to-class mistakes ordered by frequency."""

    rows = []

    # Walk through every true/predicted category combination.
    for actual_category in confusion_df.index:
        for predicted_category in confusion_df.columns:
            # Diagonal cells are correct predictions, not confusion errors.
            if actual_category == predicted_category:
                continue

            count = confusion_df.loc[
                actual_category,
                predicted_category,
            ]

            # Ignore pairs that never occurred.
            if count == 0:
                continue

            rows.append(
                {
                    "actual_category": actual_category,
                    "predicted_category": predicted_category,
                    "count": int(count),
                }
            )

    # Highest-frequency errors should appear first.
    return (
        pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    )


def get_top_features_by_class(
    model,
    top_n: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """Return terms with the strongest positive weights per class."""

    # Access fitted pipeline stages.
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    # Feature positions correspond to coefficient columns.
    feature_names = vectorizer.get_feature_names_out()

    results = {}

    # Each coefficient row corresponds to one classifier class.
    for class_index, class_name in enumerate(classifier.classes_):
        class_coefficients = classifier.coef_[class_index]

        # Get indexes of the largest positive coefficients.
        top_indexes = np.argsort(class_coefficients)[-top_n:][::-1]

        results[class_name] = [
            (
                feature_names[index],
                float(class_coefficients[index]),
            )
            for index in top_indexes
        ]

    return results


import pandas as pd


def build_error_analysis(
    dataframe: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Build a table containing only incorrect predictions."""

    # Generate predicted class labels.
    predictions = model.predict(dataframe["text"])

    # Generate per-class probability estimates.
    probabilities = model.predict_proba(dataframe["text"])

    # The largest probability belongs to the selected prediction.
    confidence = probabilities.max(axis=1)

    analysis = dataframe[
        [
            "ticket_id",
            "text",
            "category",
        ]
    ].copy()

    analysis = analysis.rename(
        columns={
            "category": "actual_category",
        }
    )

    analysis["predicted_category"] = predictions
    analysis["confidence"] = confidence

    # Keep only misclassified records.
    errors = analysis[
        analysis["actual_category"] != analysis["predicted_category"]
    ].copy()

    # Highest-confidence errors are often worth reviewing first.
    return errors.sort_values(
        "confidence",
        ascending=False,
    ).reset_index(drop=True)
