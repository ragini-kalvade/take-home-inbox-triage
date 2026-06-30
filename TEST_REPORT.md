# Test Report — Inbox Triage Skill

This report demonstrates how I built confidence in the skill. **No LLM API key
and no running server are required to reproduce it** — the suite mocks the HTTP
client and the LLM, so the routing, security, fail-safe logic, and metrics
aggregators are verified deterministically and offline.

## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test
```

## Result

`44 passed` — full run captured below (Python 3.14.2, pytest 9.1.1).

```
============================== 44 passed in 0.69s ==============================
```

## What each group of tests proves

| Requirement | Tests |
|---|---|
| **Correct routing** (billing→reply, bug→alert, sales→reply+lead, spam→none) | `test_plan_actions_*` |
| **Human-in-the-loop gate** — nothing executes without approval | `test_execute_denied_*`, `test_triage_inbox_denied_*`, `test_triage_inbox_propose_mode_*` |
| **Least privilege** — spam/denied paths never request the write token | `test_triage_inbox_spam_*`, `test_triage_client_write_without_token` |
| **Approved writes dispatch** to the right endpoint | `test_execute_approved_*`, `test_triage_inbox_approved_*` |
| **Read scope uses the read token only** | `test_get_inbox_uses_read_token` |
| **Fail-safe classification** — invalid LLM output falls back to `spam` | `test_classify_fallback_*`, `test_parse_label_*` |
| **LLM provider routing** | `test_classify_via_groq_provider`, `test_llm_provider_rejects_unknown` |
| **Resilience** — one bad email doesn't abort the run | `test_triage_inbox_one_email_error_continues` |
| **Tier A classification metrics** — gold labels, spam FN/FP, prompt-injection | `test_load_expected_labels`, `test_classification_*`, `test_prompt_injection_caught` |
| **Tier A human gate** — approval without edit | `test_gate_metrics_*` |
| **Tier A safety invariants** — write token = executes, no spam writes | `test_safety_invariants_*`, `test_compute_run_metrics_*` |
| **Tier A funnel** — partial completion on multi-action emails | `test_funnel_partial_completion` |
| **Tier B draft quality** — `score_draft_quality`, `draft_source=template` | `test_score_draft_quality_*`, `test_draft_source_on_plan_actions` |

## Manual / live verification (optional, needs a key)

With a running `make serve` and an LLM key configured in `.env`:

1. `python -m src.triage_skill` (propose-only) → `make audit` shows **no** side effects; run ends with **Run metrics** including classification accuracy vs gold labels.
2. `python -m src.triage_skill --approve` → approve actions with `[y/N/e]`; metrics show approval rate, safety invariants, and draft pass rate.

The skill supports two LLM providers via `LLM_PROVIDER` (defaults to
`anthropic`; set to `groq` for Groq). The mocked suite above runs identically
with no key.
