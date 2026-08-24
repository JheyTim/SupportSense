# TF-IDF converts ticket text into sparse numeric features.
from sklearn.feature_extraction.text import TfidfVectorizer

# Import each classifier we want to benchmark.
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_logistic_regression() -> LogisticRegression:
    """Create the Logistic Regression configuration used in experiments."""

    # Keep parameters explicit so experiment results are reproducible.
    return LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)


def build_multinomial_nb() -> MultinomialNB:
    """Create a Multinomial Naive Bayes classifier."""

    # Start with the library's conventional alpha=1.0 smoothing.
    return MultinomialNB(alpha=1.0)


def build_complement_nb() -> ComplementNB:
    """Create a Complement Naive Bayes classifier."""

    # Begin with the default smoothing strength.
    return ComplementNB(alpha=1.0)


def build_linear_svc() -> LinearSVC:
    """Create a linear support-vector classifier."""

    # C controls regularization strength inversely.
    return LinearSVC(C=1.0, max_iter=5000)


def build_text_pipeline(
    classifier,
    *,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: float = 1,
    max_df: float = 1.0,
    max_features: int | None = None,
    sublinear_tf: bool = False,
) -> Pipeline:
    """Create a configurable TF-IDF text classification pipeline."""

    # Feature extraction remains inside the pipeline so it is fitted
    # only on the training dataset.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )

    # Keeping vectorization and classification together helps prevent
    # accidentally fitting text features against validation/test data.
    return Pipeline(steps=[("tfidf", vectorizer), ("classifier", classifier)])


def create_classifier(name: str):
    """Create a classifier from an experiment configuration."""

    # Keep classifier creation in one location so configurations remain
    # consistent across experiments.
    if name == "logistic_regression":
        return build_logistic_regression()

    if name == "multinomial_nb":
        return build_multinomial_nb()

    if name == "complement_nb":
        return build_complement_nb()

    if name == "linear_svc":
        return build_linear_svc()

    raise ValueError(f"Unsupported classifier: {name}")
