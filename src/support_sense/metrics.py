from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def calculate_classification_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """Calculate the baseline multiclass evaluation metrics."""

    # Accuracy measures the overall fraction classified correctly.
    accuracy = accuracy_score(y_true, y_pred)

    # Macro averaging gives every category equal importance.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }
