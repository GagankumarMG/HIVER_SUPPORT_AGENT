import pytest

from src.intents.classifier import TfidfLogisticClassifier


TRAIN_TEXTS = [
    "my payment failed", "I was charged twice", "payment did not go through",
    "please cancel my subscription", "cancel my order now", "I want to cancel",
    "cannot log into my account", "forgot my password", "locked out of account",
] * 3
TRAIN_LABELS = (
    ["PAYMENT_ISSUE"] * 3 + ["CANCELLATION"] * 3 + ["LOGIN_OR_ACCOUNT"] * 3
) * 3


def test_tfidf_classifier_predicts_known_label():
    clf = TfidfLogisticClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    pred = clf.predict("my payment failed again", confidence_threshold=0.0)
    assert pred.intent in {"PAYMENT_ISSUE", "CANCELLATION", "LOGIN_OR_ACCOUNT"}
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_k) <= 3


def test_tfidf_classifier_below_threshold_returns_other():
    clf = TfidfLogisticClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    pred = clf.predict("completely unrelated gibberish text zzqx", confidence_threshold=0.99)
    assert pred.intent == "OTHER_OR_UNKNOWN"
