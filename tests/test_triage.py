"""Unit tests for inbox triage skill."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.triage_skill import (
    LABELS,
    ActionOutcome,
    ApprovalDecision,
    Classification,
    GROQ_URL,
    ProposedAction,
    TriageClient,
    TriageResult,
    _classify,
    _coerce_approval,
    _edit_distance,
    _llm_provider,
    _parse_label_from_text,
    _reply_body,
    compute_classification_metrics,
    compute_funnel_metrics,
    compute_gate_metrics,
    compute_run_metrics,
    compute_safety_metrics,
    execute,
    load_expected_labels,
    plan_actions,
    score_draft_quality,
    score_reply_template,
    triage_inbox,
)

SAMPLE_EMAIL = {
    "id": "e-001",
    "from": "dana.whitfield@meridianparts.com",
    "subject": "Invoice #4471 charged twice this month",
    "received_at": "2026-06-22T08:14:00Z",
    "body": "We were billed twice for invoice #4471.",
}

BUG_EMAIL = {
    "id": "e-002",
    "from": "marcus@brightlee.io",
    "subject": "Export to CSV silently drops the last row",
    "received_at": "2026-06-22T09:02:00Z",
    "body": "CSV only contains N-1 rows.",
}

SALES_EMAIL = {
    "id": "e-003",
    "from": "priya.n@northwind-logistics.com",
    "subject": "Interested in a pilot for our ops team",
    "received_at": "2026-06-22T10:41:00Z",
    "body": "We'd love to explore a pilot.",
}

SPAM_EMAIL = {
    "id": "e-004",
    "from": "winner@lucky-rewards-intl.biz",
    "subject": "YOU have been SELECTED",
    "received_at": "2026-06-22T11:20:00Z",
    "body": "Click here to claim your prize.",
}


def test_plan_actions_billing():
    actions = plan_actions("billing", SAMPLE_EMAIL)
    assert len(actions) == 1
    assert actions[0].kind == "send_reply"
    assert actions[0].payload["to"] == SAMPLE_EMAIL["from"]
    assert actions[0].payload["subject"] == f"Re: {SAMPLE_EMAIL['subject']}"
    assert actions[0].payload["in_reply_to"] == SAMPLE_EMAIL["id"]
    assert "billing" in actions[0].payload["body"].lower() or "review" in actions[0].payload["body"].lower()


def test_plan_actions_bug_report():
    actions = plan_actions("bug_report", BUG_EMAIL)
    assert len(actions) == 1
    assert actions[0].kind == "send_alert"
    assert actions[0].payload["channel"] == "#engineering"
    assert BUG_EMAIL["subject"] in actions[0].payload["message"]


def test_plan_actions_sales_lead():
    actions = plan_actions("sales_lead", SALES_EMAIL)
    assert len(actions) == 2
    assert actions[0].kind == "send_reply"
    assert actions[1].kind == "create_lead"
    assert actions[1].payload["email"] == SALES_EMAIL["from"]
    assert actions[1].payload["name"] == "priya.n"
    assert actions[1].payload["summary"] == SALES_EMAIL["subject"]
    assert actions[1].payload["company"] == "Northwind Logistics"


def test_company_inference_skips_free_providers():
    email = {**SALES_EMAIL, "from": "jane.doe@gmail.com"}
    lead = plan_actions("sales_lead", email)[1]
    assert "company" not in lead.payload


def test_company_inference_from_corporate_domain():
    email = {**SALES_EMAIL, "from": "ops@acme-corp.io"}
    lead = plan_actions("sales_lead", email)[1]
    assert lead.payload["company"] == "Acme Corp"


def test_company_inference_collapses_subdomains():
    for addr in ("ops@acme.com", "ops@mail.acme.com", "ops@eu.corp.acme.com"):
        email = {**SALES_EMAIL, "from": addr}
        lead = plan_actions("sales_lead", email)[1]
        assert lead.payload["company"] == "Acme", addr


def test_company_inference_handles_compound_suffix():
    email = {**SALES_EMAIL, "from": "ops@acme.co.uk"}
    lead = plan_actions("sales_lead", email)[1]
    assert lead.payload["company"] == "Acme"


def test_plan_actions_spam():
    assert plan_actions("spam", SPAM_EMAIL) == []


def test_reply_body_reply_labels():
    for label in ("billing", "bug_report", "sales_lead"):
        assert _reply_body(label).strip()


def test_reply_body_spam_raises():
    with pytest.raises(ValueError, match="No reply template"):
        _reply_body("spam")


def test_triage_client_write_without_token():
    client = TriageClient("http://test", read_token="read")
    with pytest.raises(RuntimeError, match="Write scope required"):
        client.send_reply(to="a@b.com", subject="Hi", body="Hello")


def test_execute_denied_never_calls_provider():
    provider = MagicMock()
    action = ProposedAction(kind="send_reply", payload={})
    result = execute(
        action,
        base_url="http://test",
        write_token_provider=provider,
        approved=False,
        read_token="read",
    )
    assert result is None
    provider.assert_not_called()


def test_execute_approved_calls_provider_and_dispatches():
    provider = MagicMock(return_value="write-token")
    action = ProposedAction(
        kind="send_alert",
        payload={"channel": "#engineering", "message": "bug"},
    )
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"status": "posted", "id": "alert-1"}

    with patch.object(httpx.Client, "post", return_value=mock_response) as mock_post:
        result = execute(
            action,
            base_url="http://127.0.0.1:8099",
            write_token_provider=provider,
            approved=True,
            read_token="read-token",
        )

    provider.assert_called_once()
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[0][0] == "http://127.0.0.1:8099/slack/alert"
    assert call_kwargs[1]["json"] == {"channel": "#engineering", "message": "bug"}
    assert result == {"status": "posted", "id": "alert-1"}


def test_triage_inbox_spam_never_calls_write_provider():
    read_client = MagicMock()
    read_client.read_token = "read"
    read_client.get_inbox.return_value = [SPAM_EMAIL]
    provider = MagicMock()

    results = triage_inbox(
        read_client,
        approver=lambda e, a: True,
        classifier=lambda e: "spam",
        mode="approve",
        base_url="http://test",
        write_token_provider=provider,
    )

    assert len(results) == 1
    assert results[0].label == "spam"
    assert results[0].actions == []
    provider.assert_not_called()


def test_triage_inbox_denied_never_calls_write_provider():
    read_client = MagicMock()
    read_client.read_token = "read"
    read_client.get_inbox.return_value = [SAMPLE_EMAIL]
    provider = MagicMock()

    results = triage_inbox(
        read_client,
        approver=lambda e, a: False,
        classifier=lambda e: "billing",
        mode="approve",
        base_url="http://test",
        write_token_provider=provider,
    )

    assert len(results) == 1
    assert results[0].skipped == ["send_reply"]
    assert results[0].approved == []
    provider.assert_not_called()


def test_triage_inbox_approved_executes_write():
    read_client = MagicMock()
    read_client.read_token = "read"
    read_client.get_inbox.return_value = [SAMPLE_EMAIL]
    provider = MagicMock(return_value="write-token")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"status": "sent", "id": "mail-1"}

    with patch.object(httpx.Client, "post", return_value=mock_response):
        results = triage_inbox(
            read_client,
            approver=lambda e, a: True,
            classifier=lambda e: "billing",
            mode="approve",
            base_url="http://127.0.0.1:8099",
            write_token_provider=provider,
        )

    assert len(results) == 1
    assert results[0].approved == ["send_reply"]
    assert len(results[0].executed) == 1
    provider.assert_called_once()


def test_triage_inbox_propose_mode_skips_approver():
    read_client = MagicMock()
    read_client.read_token = "read"
    read_client.get_inbox.return_value = [SAMPLE_EMAIL]
    approver = MagicMock(return_value=True)
    provider = MagicMock()

    results = triage_inbox(
        read_client,
        approver,
        classifier=lambda e: "billing",
        mode="propose",
        base_url="http://test",
        write_token_provider=provider,
    )

    assert len(results) == 1
    assert len(results[0].actions) == 1
    approver.assert_not_called()
    provider.assert_not_called()


def test_classify_fallback_on_invalid_json():
    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="not valid json at all")]
        )
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            result = _classify(SAMPLE_EMAIL)

    assert result.label == "spam"
    assert result.error is True


def test_parse_label_from_json():
    assert _parse_label_from_text('{"label": "billing"}') == "billing"


def test_parse_label_rejects_prose_substring():
    assert _parse_label_from_text("This is not billing. It is spam.") is None


def test_parse_label_rejects_invalid_label():
    assert _parse_label_from_text('{"label": "urgent"}') is None


def test_plan_actions_missing_field_raises():
    with pytest.raises(ValueError, match="missing required email field"):
        plan_actions("billing", {"id": "x", "subject": "hi"})


def test_classify_valid_response():
    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"label": "bug_report"}')]
        )
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            result = _classify(BUG_EMAIL)

    assert result == Classification(label="bug_report", error=False)


def test_triage_inbox_one_email_error_continues():
    read_client = MagicMock()
    read_client.read_token = "read"
    read_client.get_inbox.return_value = [
        {"id": "bad", "from": "a@b.com", "subject": "x", "body": "y"},
        SAMPLE_EMAIL,
    ]

    def flaky_classifier(email: dict) -> str:
        if email["id"] == "bad":
            raise RuntimeError("classifier blew up")
        return "billing"

    results = triage_inbox(
        read_client,
        approver=None,
        classifier=flaky_classifier,
        mode="propose",
        base_url="http://test",
    )

    assert len(results) == 2
    assert results[0].email_id == "bad"
    assert results[0].classification_error is True
    assert results[1].email_id == SAMPLE_EMAIL["id"]
    assert results[1].label == "billing"


def test_get_inbox_uses_read_token():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = [SAMPLE_EMAIL]

    with patch.object(httpx.Client, "get", return_value=mock_response) as mock_get:
        client = TriageClient("http://127.0.0.1:8099", read_token="read-token")
        inbox = client.get_inbox()
        client.close()

    assert inbox == [SAMPLE_EMAIL]
    mock_get.assert_called_once()
    headers = mock_get.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer read-token"


def test_classify_via_groq_provider():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"label": "sales_lead"}'}}]
    }

    with patch("src.triage_skill.httpx.post", return_value=mock_response) as mock_post:
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test-key"},
            clear=False,
        ):
            result = _classify(SALES_EMAIL)

    assert result == Classification(label="sales_lead", error=False)
    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == GROQ_URL
    assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer test-key"


def test_llm_provider_rejects_unknown():
    with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=False):
        with pytest.raises(ValueError, match="Invalid LLM_PROVIDER"):
            _llm_provider()


def test_score_reply_template_passes_for_billing():
    score = score_reply_template("billing", _reply_body("billing"))
    assert score.passed
    assert score.checks["label_relevant"]


def test_score_reply_template_fails_on_forbidden_phrase():
    score = score_reply_template("billing", "We guarantee a refund $500 within 24 hours.")
    assert not score.passed
    assert not score.checks["no_forbidden"]


def test_compute_run_metrics_propose_mode():
    results = [
        TriageResult(
            email_id="e-001",
            label="billing",
            actions=plan_actions("billing", SAMPLE_EMAIL),
        ),
        TriageResult(email_id="e-004", label="spam", actions=[]),
    ]
    metrics = compute_run_metrics(results, mode="propose")
    assert metrics.emails == 2
    assert metrics.by_label == {"billing": 1, "spam": 1}
    assert metrics.actions_proposed == 1
    assert metrics.approval_rate is None
    assert metrics.draft_pass_rate == 1.0
    assert metrics.template_pass_rate == 1.0
    assert metrics.safety is not None
    assert metrics.safety.invariants_ok


def test_compute_run_metrics_approve_mode():
    results = [
        TriageResult(
            email_id="e-001",
            label="billing",
            actions=plan_actions("billing", SAMPLE_EMAIL),
            approved=["send_reply"],
            executed=[{"status": "sent"}],
            outcomes=[
                ActionOutcome(
                    kind="send_reply",
                    approved=True,
                    executed=True,
                    review_seconds=1.0,
                )
            ],
        ),
        TriageResult(
            email_id="e-002",
            label="bug_report",
            actions=plan_actions("bug_report", BUG_EMAIL),
            skipped=["send_alert"],
            outcomes=[
                ActionOutcome(kind="send_alert", review_seconds=0.5),
            ],
        ),
    ]
    metrics = compute_run_metrics(results, mode="approve", write_token_accesses=1)
    assert metrics.actions_proposed == 2
    assert metrics.actions_approved == 1
    assert metrics.actions_skipped == 1
    assert metrics.approval_rate == pytest.approx(1 / 2)
    assert metrics.draft_pass_rate == 1.0
    assert metrics.safety is not None
    assert metrics.safety.invariants_ok
    assert metrics.gate is not None
    assert metrics.gate.approval_without_edit_rate == 1.0


def test_load_expected_labels():
    gold = load_expected_labels()
    assert gold is not None
    assert gold["e-001"] == "billing"
    assert set(gold["e-008"]) == {"billing", "sales_lead"}


def test_classification_metrics_accuracy():
    gold = {"e-001": "billing", "e-004": "spam", "e-008": ["billing", "sales_lead"]}
    results = [
        TriageResult(email_id="e-001", label="billing", actions=[]),
        TriageResult(email_id="e-004", label="spam", actions=[]),
        TriageResult(email_id="e-008", label="sales_lead", actions=[]),
    ]
    m = compute_classification_metrics(results, gold)
    assert m.accuracy == 1.0
    assert m.spam_false_negatives == 0
    assert m.spam_false_positives == 0


def test_classification_spam_false_negative():
    gold = {"e-004": "spam"}
    results = [
        TriageResult(
            email_id="e-004",
            label="billing",
            actions=plan_actions("billing", SPAM_EMAIL),
        ),
    ]
    m = compute_classification_metrics(results, gold)
    assert m.spam_false_negatives == 1


def test_spam_false_positive():
    gold = {"e-001": "billing"}
    results = [TriageResult(email_id="e-001", label="spam", actions=[])]
    m = compute_classification_metrics(results, gold)
    assert m.spam_false_positives == 1


def test_prompt_injection_caught():
    gold = {"e-007": "spam"}
    results = [TriageResult(email_id="e-007", label="spam", actions=[])]
    m = compute_classification_metrics(results, gold)
    assert m.prompt_injection_caught is True


def test_gate_metrics_approval_without_edit():
    results = [
        TriageResult(
            email_id="e-001",
            label="billing",
            actions=plan_actions("billing", SAMPLE_EMAIL),
            approved=["send_reply"],
            outcomes=[
                ActionOutcome(kind="send_reply", approved=True, edited=False, review_seconds=1.0),
            ],
        ),
    ]
    g = compute_gate_metrics(results)
    assert g.approval_without_edit_rate == 1.0


def test_edit_distance_counts_same_length_replacements():
    assert _edit_distance("abc", "xyz") == 3
    assert _edit_distance("kitten", "sitting") == 3


def test_safety_invariants_ok():
    results = [
        TriageResult(
            email_id="e-001",
            label="billing",
            approved=["send_reply"],
            executed=[{"ok": True}],
        ),
    ]
    s = compute_safety_metrics(results, mode="approve", write_token_accesses=1)
    assert s.invariants_ok


def test_safety_invariants_fail_spam_write():
    results = [
        TriageResult(email_id="e-004", label="spam", approved=["x"], executed=[{"ok": True}]),
    ]
    s = compute_safety_metrics(results, mode="approve", write_token_accesses=1)
    assert not s.invariants_ok
    assert s.writes_on_spam == 1


def test_funnel_partial_completion():
    results = [
        TriageResult(
            email_id="e-003",
            label="sales_lead",
            approved=["send_reply"],
            skipped=["create_lead"],
        ),
    ]
    f = compute_funnel_metrics(results)
    assert f.partial_completion.get("reply_without_lead") == 1


def test_draft_source_on_plan_actions():
    actions = plan_actions("billing", SAMPLE_EMAIL)
    assert actions[0].draft_source == "template"
    assert actions[0].draft_model is None


def test_score_draft_quality_alias():
    assert score_draft_quality("billing", _reply_body("billing")).passed


def test_coerce_approval_bool():
    assert _coerce_approval(True).approved
    assert not _coerce_approval(False).approved


def test_coerce_approval_decision():
    d = ApprovalDecision(approved=True, final_body="Hi", review_seconds=0.2)
    assert _coerce_approval(d) is d
