# Engineering Manager's Log

> One page. This is where you show us how you *directed* the AI — it matters as much
> as the code. Be concrete. Bullet points are fine.

**Name:** Ragini Kalvade
**Time spent (be honest):** ~2.5 hours

---

## How I broke the work down

1. Read `mock_api/server.py` and `env.example` first to lock payload field names and auth rules.
2. Implemented pure routing (`ROUTING` + `plan_actions`) — testable with no network.
3. Built `TriageClient` against exact Pydantic schemas (`to`, `subject`, `body`, `in_reply_to`, etc.).
4. Added `_classify` / `classify_email` with Anthropic JSON-only output and spam fallback.
5. Changed `execute` signature to defer `WRITE_TOKEN` until after per-action approval.
6. Wired `triage_inbox` with explicit `mode="propose" | "approve"`.
7. Added CLI, pytest suite, README section, and manual `make audit` smoke path.
8. Polished approval UI in the final pass: per-email context block (from, body snippet, draft reply preview), full draft body on each `--approve` prompt, and EOF-safe deny.
9. **Metrics (Tier A/B):** end-of-run dashboard with classification accuracy vs `fixtures/expected_labels.json`, human-gate stats (approval w/o edit, review time), safety invariants (write-token accesses = executes), funnel partial-completion, and v2-ready `draft_source` + `score_draft_quality` on proposed replies.

## Where I ran things in parallel

- Cross-verification plan review (architecture + fixture expectations) while reading the stub signatures.
- Test cases for `plan_actions` drafted alongside routing table fill-in (pure functions first).

## One time the AI was wrong, and how I caught it

The initial plan said "keep stub signatures" but also changed `execute` to use `write_token_provider` instead of passing a prebuilt client. That would have left `WRITE_TOKEN` reachable earlier in the run if we followed the stub literally. I caught the contradiction during cross-verification against the repo and prioritized the security boundary: build the write-capable client **inside** `execute` only when `approved=True`.

## What I deliberately cut to fit the 2 hours

- LLM-generated reply prose → deterministic templates (safer, faster to test).
- Retry/backoff, circuit breakers, cross-run idempotency.
- Full response-shape validation beyond `raise_for_status()`.
- CI / Docker (pytest works via `make test`).

**Tradeoff:** rerunning with `--approve` creates new side effects; mock audit resets on server restart. Restart the server before a clean Loom demo if you ran approve earlier.

## Known limitations

- **CRM `company` inference is a heuristic, not identity resolution.** It infers the company from the sender's registrable domain (subdomains like `mail.acme.com` collapse to `Acme`). It *cannot* unify genuinely unrelated domains for the same company (e.g. `acme.com` vs `acme.io`) — that belongs to CRM-side dedup or the human approver at the approval gate.
- **`_COMPOUND_SUFFIXES` is a pragmatic shortlist** (`co.uk`, `com.au`, …) rather than the full public-suffix list. Full correctness would mean pulling in `tldextract`, which felt like overkill for the take-home scope.

## The design decision I'm proudest of

The safety boundary between planning and doing. The agent can classify an email and create structured proposed actions, but those proposals cannot send emails, create CRM records, or post alerts on their own. A human must approve each action first. Spam gets dropped before approval, propose-only mode never reads the write token, and denied actions stop before any write credentials are accessed. I also changed `execute` to receive a `write_token_provider` instead of a prebuilt write-capable client, so tests can prove that unsafe paths never touch write credentials.

