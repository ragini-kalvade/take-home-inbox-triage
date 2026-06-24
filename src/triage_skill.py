"""Inbox Triage skill worker — STUB.

This is where you work. The signatures below are a suggested starting shape —
keep them, change them, or add to them as you see fit. Replace every
`raise NotImplementedError` with a real implementation.

You are free to choose how you classify emails (an LLM call is the obvious move —
that's the point of the role), how you structure the human-in-the-loop gate, and
how you wire the client. The requirements are in the README; how you interpret and
verify "done" is part of what we're looking at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The only four labels a triage may produce.
LABELS = ("billing", "bug_report", "sales_lead", "spam")

# Which actions each classification implies. `spam` implies none.
# (Filling this in correctly is part of the task — it is intentionally empty.)
ROUTING: dict[str, list[str]] = {
    "billing": [],
    "bug_report": [],
    "sales_lead": [],
    "spam": [],
}

# Action kinds your plan may contain.
ACTION_KINDS = ("send_reply", "send_alert", "create_lead")


@dataclass
class ProposedAction:
    """An action the agent WANTS to take. Proposing is not doing — nothing here
    touches the outside world until it has been approved and executed."""

    kind: str
    payload: dict
    # Every external write requires the write scope. Reads/no-ops do not.
    requires_write: bool = True
    rationale: str = ""


@dataclass
class TriageResult:
    email_id: str
    label: str
    actions: list[ProposedAction] = field(default_factory=list)


class TriageClient:
    """Thin wrapper over the mock API. Implement the HTTP calls.

    Construct it with the base URL and the tokens it is allowed to use. Think
    about which methods need which scope.
    """

    def __init__(self, base_url: str, read_token: str, write_token: str | None = None):
        raise NotImplementedError

    def get_inbox(self) -> list[dict]:
        raise NotImplementedError

    def send_reply(self, *, to: str, subject: str, body: str, in_reply_to: str | None = None) -> dict:
        raise NotImplementedError

    def send_alert(self, *, channel: str, message: str) -> dict:
        raise NotImplementedError

    def create_lead(self, *, name: str, email: str, company: str | None = None, summary: str | None = None) -> dict:
        raise NotImplementedError


def classify_email(email: dict) -> str:
    """Return exactly one of LABELS for the given email.

    This is the obvious place to use AI. Whatever you do, the return value must
    always be a member of LABELS.
    """
    raise NotImplementedError


def plan_actions(label: str, email: dict) -> list[ProposedAction]:
    """Turn a classification into the actions it implies, per the routing table.

    Pure and deterministic — no network, no LLM, no side effects. `spam` plans
    nothing.
    """
    raise NotImplementedError


def execute(action: ProposedAction, client: TriageClient, *, approved: bool) -> dict | None:
    """Execute a single proposed action — but only if a human approved it.

    This is the human-in-the-loop gate. If `approved` is False, NOTHING external
    may happen: return None and do not call the client.

    Contract: dispatch on `action.kind`; `action.payload` holds the keyword
    arguments for the matching client method (e.g. a `send_reply` action calls
    `client.send_reply(**action.payload)`).
    """
    raise NotImplementedError


def triage_inbox(client: TriageClient, approver, classifier=classify_email) -> list[TriageResult]:
    """Orchestrate the whole run: fetch the inbox, classify each email, plan
    actions, ask `approver` to approve each proposed action, and execute only the
    approved ones.

    `approver` is a callable: `approver(email, action) -> bool`. (In production
    this would surface a human-in-the-loop card; in tests it is a stub.)

    `classifier` is injectable so the orchestration can be tested without a live
    model. It defaults to `classify_email`.

    Return one TriageResult per email.
    """
    raise NotImplementedError
