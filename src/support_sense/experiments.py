from time import perf_counter

from support_sense.metrics import (
    calculate_classification_metrics,
)


def run_experiment(
    *, experiment_id: str, pipeline, X_train, y_train, X_validation, y_validation
) -> tuple[dict, object]:
    """Fit one model experiment and return metrics plus the fitted model."""

    # Start immediately before training so feature fitting and classifier
    # fitting are both included in the elapsed time.
    start_time = perf_counter()

    pipeline.fit(X_train, y_train)

    training_seconds = perf_counter() - start_time

    # Validation data must only pass through already-fitted pipeline steps.
    predictions = pipeline.predict(X_validation)

    # Reuse exactly the same metrics as previous milestones.
    metrics = calculate_classification_metrics(y_validation, predictions)

    # Record the number of text features learned from training data.
    vectorizer = pipeline.named_steps["tfidf"]

    vocabulary_size = len(vectorizer.get_feature_names_out())

    result = {
        "experiment_id": experiment_id,
        "training_seconds": training_seconds,
        "vocabulary_size": vocabulary_size,
        **metrics,
    }

    return result, pipeline
