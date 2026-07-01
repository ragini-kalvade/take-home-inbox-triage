# Go Fig — AI Engineer · Project Take-Home

**Inbox Triage Agent**

---

## The rules

- **Time cap: 2 hours.** Pick a single uninterrupted block. A clean, working *core* beats a
  sprawling unfinished pile — and we mean the cap. (Suggested split below.)
- **Use AI heavily.** This is the job. Cursor, Claude Code, whatever you run day-to-day.
  We are **not** testing whether you can hand-write Python. We're testing how well you
  *direct* AI to build correct, secure software under a deadline. Treat the AI like a team
  of engineers you're managing.
- We explicitly do **not** penalize AI use. We reward *managed* AI use.
- **"Done" is yours to define.** There's no hidden test suite grading you to a spec. We've
  left room on purpose — show us your judgment about what matters and where to spend effort.

## How to spend your two hours

| Time | Focus |
|---|---|
| **~60 min** | **Build** the skill against the requirements below. |
| **~30 min** | **Test / verify** it however you see fit — make sure it actually works. |
| **~30 min** | **Wrap up the deliverables** — clean up the repo, fill in the engineering log, record your Loom. |

Budget for the wrap-up; don't let it get squeezed. We care as much about how you finish and
communicate as about the code itself.

## The scenario

A client — a small B2B company — wants an agent that triages their incoming customer
emails so a human never starts from a blank page. You're building the first skill worker.

This repo is a scaffold: a mock REST API (inbox + outbound mail + CRM), email fixtures,
env config, and a **stubbed skill module**. Build the skill.

> **You need no external accounts.** The mock API stands in for Gmail and the CRM — it runs
> locally with `make serve`. The only thing you bring is your own LLM API key.

## Requirements

1. **Ingest** the incoming emails from the mock `GET /inbox` endpoint.
2. **Classify** each email into exactly one of: `billing`, `bug_report`, `sales_lead`, `spam`.
3. **Draft an action** per the routing table:

   | Classification | Action |
   |---|---|
   | `billing` | draft a reply to the customer (`POST /mail/send`) |
   | `bug_report` | alert the engineering team (`POST /slack/alert`, channel `#engineering`) |
   | `sales_lead` | draft a reply **and** create a CRM lead (`POST /mail/send`, `POST /crm/lead`) |
   | `spam` | no action — log and drop |

4. **Human-in-the-loop gate.** *No external action (send reply, create CRM record) may
   execute without explicit human approval.* The skill **proposes**, a human **approves**,
   and only then does it call the write endpoint. Design this gate.
5. **Least privilege & secrets.** The spam path must never hold write credentials. All
   tokens come from the environment — never hardcoded. The write scope is used only after
   approval.
6. **Verify your work.** How you prove it works — tests, a demo script, manual checks — is
   up to you. We want to see how you build confidence in your own output.
7. **README the client could read.** Append a short section below: what it does, how to
   run it, and the one design decision you're proudest of.

## What we hand you

```
mock_api/server.py     FastAPI mock: /inbox, /mail/send, /slack/alert, /crm/lead
fixtures/emails.json   the inbox the agent triages
src/triage_skill.py    inbox triage skill (classify, route, propose/approve)
env.example            the env vars you need (copy to .env)
Makefile               `make serve` (run the API), `make audit` (inspect side effects)
ENGINEERING_LOG.md     a one-page template — fill it in
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env           # then fill in your own LLM API key (any provider)
make serve                    # terminal 1 — starts the mock API on :8099
```

## Deliverables (submit all three)

1. **A link to your GitHub repo.** Fork this repo, push your edits, and share the URL
   with us. (Public, or private with us added as collaborators — your call.)
2. **`ENGINEERING_LOG.md`**, filled in (one page) — how you directed the work.
3. **A Loom recording (required, ≤5 min).** Walk us through what you built, demo it
   running, and call out a decision or two you're proud of. This is where we see your
   communication and how completely you finished — treat it like showing a client.

## How we evaluate

We grade *how you managed the AI* as much as the result: did you decompose and delegate,
review its output critically, catch its mistakes, and make sound security calls? We also
look at how you **interpreted an open-ended problem** and how clearly you **communicate**
your work. The full rubric is shared with you after you submit.

Questions before you start? Email us. Once you open the scaffold, the clock is yours.

---

<!-- ↓↓↓ CANDIDATE: add your "README the client could read" section here ↓↓↓ -->

## Inbox Triage Skill — What We Built

This skill reads your incoming customer emails, classifies each one, and **proposes** the right next actions — reply, engineering alert, or CRM lead — so your team never starts from a blank page.

The skill **drafts/proposes** replies locally using safe templates. The mock send endpoint (`POST /mail/send`) is called **only after you explicitly approve** each action.

### How it works

1. **Ingest** — fetches all emails from the mock inbox (`GET /inbox`).
2. **Classify** — uses the configured LLM provider to assign each email one label: `billing`, `bug_report`, `sales_lead`, or `spam`.
3. **Plan** — deterministic Python routing decides what to do (the LLM never picks endpoints). Bug reports trigger an engineering alert; spam is logged and dropped with no reply template.
4. **Propose or approve** — by default the skill prints proposed actions only. With `--approve`, you confirm each action individually before anything is sent.
5. **Execute** — approved actions call the write API (mail, Slack alert, or CRM lead).

### How to run

```bash
cp env.example .env          # add your LLM key
make serve                   # terminal 1 — mock API on :8099
python -m src.triage_skill    # propose-only; no writes
python -m src.triage_skill --approve   # per-action y/n prompts; writes after approval
make audit                   # inspect side effects
```

**LLM provider.** Defaults to Anthropic — set `ANTHROPIC_API_KEY`. For Groq
instead, set `LLM_PROVIDER=groq` and `GROQ_API_KEY` (optional `GROQ_MODEL`,
default `llama-3.1-8b-instant`). Groq has a no-card free tier, handy for a
live demo.

Classification is the only LLM call, so the provider swap is fully contained.

**No key? Verify offline.** `pytest -v` mocks the LLM and HTTP client, so the
routing, approval gate, and least-privilege logic are verified with no API key
and no server. See `TEST_REPORT.md` for a captured run.

**Propose-only (default):** loads `API_BASE_URL`, `READ_TOKEN`, and the selected LLM key only. `WRITE_TOKEN` is never read.

**Approve mode:** `WRITE_TOKEN` is accessed only when you approve a specific action — spam and denied actions never touch it.

### Verify it works

1. Run propose-only and confirm `make audit` shows empty side effects.
2. Run with `--approve`, approve a billing reply and a bug alert, then `make audit` to confirm entries appeared.
3. Run `make test` (or `pytest -v`) for automated checks — mocked, no live API
   or LLM required.

### Run metrics

Every run prints a **Run metrics** summary at the end so you can see how the
agent performed without digging through logs.

| Section | What it tells you |
|---------|-------------------|
| **Classification** | Label accuracy vs gold fixtures, spam false positives/negatives, whether prompt-injection mail (e-007) was caught |
| **Human gate** | Approval rate, approval without edit (draft good enough as-is), denial rate by action type, average review time |
| **Safety** | Write-token accesses vs actions executed, writes on spam (should be 0), whether security invariants passed |
| **Funnel** | Execution success rate, partial completion (e.g. reply approved but CRM lead skipped), errors by endpoint |
| **Draft quality** | Draft pass rate (non-empty, label-relevant, safe, right length) and draft source (`template` today; LLM in a future version) |

In **propose-only** mode you get classification and draft-quality metrics with
no writes. In **`--approve`** mode you also get human-gate, safety, and funnel
stats — including whether write credentials were used only for approved actions.

Offline tests in `TEST_REPORT.md` cover the same metric aggregators with no API
key required.

### Classification fallback tradeoff

If the LLM fails or returns invalid output, the email is classified as `spam` with `classification_error=True` and logged loudly. This is **safe** (no writes on uncertain mail) but may **drop legitimate email** — a deliberate fail-safe for a triage agent with write access.

### Design decision I'm proudest of

I separated proposed actions from executed actions. The LLM classifies; deterministic code enforces routing, template replies, per-action approval, and credential use. `WRITE_TOKEN` is accessed only inside `execute` after approval — spam and denied actions never touch it.

### Loom Video 
https://www.loom.com/share/b5e4490bc8f249f6a6b083a5480d655a 
