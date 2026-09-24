from src.escalation.policy import decide_escalation
from src.escalation import reason_codes as rc


BASE_KWARGS = dict(
    customer_message="my payment failed",
    intent="PAYMENT_ISSUE",
    intent_confidence=0.9,
    confidence_threshold=0.6,
    retrieved_top_score=0.8,
    min_hybrid_score_for_grounding=0.35,
    generation_grounded=True,
    generation_unsupported_claims=[],
    high_risk_keywords=["fraud"],
)


def test_auto_handle_when_all_signals_good():
    decision = decide_escalation(**BASE_KWARGS)
    assert decision.decision == "AUTO_HANDLE"
    assert decision.reason_code is None


def test_escalates_on_low_confidence():
    kwargs = {**BASE_KWARGS, "intent_confidence": 0.2}
    decision = decide_escalation(**kwargs)
    assert decision.decision == "ESCALATE"
    assert decision.reason_code == rc.LOW_INTENT_CONFIDENCE


def test_escalates_on_unknown_intent():
    kwargs = {**BASE_KWARGS, "intent": "OTHER_OR_UNKNOWN"}
    decision = decide_escalation(**kwargs)
    assert decision.reason_code == rc.UNKNOWN_INTENT


def test_escalates_on_no_relevant_history():
    kwargs = {**BASE_KWARGS, "retrieved_top_score": 0.1}
    decision = decide_escalation(**kwargs)
    assert decision.reason_code == rc.NO_RELEVANT_HISTORY


def test_escalates_on_high_risk_keyword():
    kwargs = {**BASE_KWARGS, "customer_message": "this is fraud, I want a lawsuit"}
    decision = decide_escalation(**kwargs)
    assert decision.reason_code == rc.HIGH_RISK_REQUEST


def test_escalates_on_sensitive_account_action():
    kwargs = {**BASE_KWARGS, "customer_message": "please delete my account now"}
    decision = decide_escalation(**kwargs)
    assert decision.reason_code == rc.SENSITIVE_ACCOUNT_ACTION


def test_escalates_on_unsupported_generation():
    kwargs = {**BASE_KWARGS, "generation_grounded": False,
              "generation_unsupported_claims": ["invented a refund amount"]}
    decision = decide_escalation(**kwargs)
    assert decision.reason_code == rc.UNSUPPORTED_INFORMATION
