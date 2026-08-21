# TF-IDF converts raw documents into numeric text features.
from sklearn.feature_extraction.text import TfidfVectorizer

# Logistic Regression will perform multiclass classification.
from sklearn.linear_model import LogisticRegression

# Pipeline keeps feature extraction and classification together.
from sklearn.pipeline import Pipeline


def build_baseline_pipeline() -> Pipeline:
    """Create the first SupportSense text-classification baseline."""

    # Keep the baseline intentionaly simple.
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=(1, 1))

    # Use a linear classifier that works directly with sparse TF-DF data.
    classifier = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)

    # The pipeline guarantees that raw text first passes through
    # TF-IDF and the resulting numeric features go to the classifier.
    return Pipeline(steps=[("tfidf", vectorizer), ("classifier", classifier)])
