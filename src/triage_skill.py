"""Inbox Triage skill worker."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx

# The only four labels a triage may produce.
LABELS = ("billing", "bug_report", "sales_lead", "spam")

# Which actions each classification implies. `spam` implies none.
ROUTING: dict[str, list[str]] = {
    "billing": ["send_reply"],
    "bug_report": ["send_reply", "send_alert"],
    "sales_lead": ["send_reply", "create_lead"],
    "spam": [],
}

# Action kinds your plan may contain.
ACTION_KINDS = ("send_reply", "send_alert", "create_lead")

HTTP_TIMEOUT = 30.0

REPLY_TEMPLATES: dict[str, str] = {
    "billing": (
        "Thank you for reaching out. We have received your billing inquiry and "
        "our team will review your billing details shortly."
    ),
    "bug_report": (
        "Thank you for reporting this issue. Our engineering team has been "
        "notified and is looking into it. We will follow up once we have an update."
    ),
    "sales_lead": (
        "Thank you for your interest. We appreciate you reaching out and "
        "we'll follow up shortly with next steps."
    ),
}


@dataclass
class ProposedAction:
    """An action the agent WANTS to take. Proposing is not doing — nothing here
    touches the outside world until it has been approved and executed."""

    kind: str
    payload: dict
    requires_write: bool = True
    rationale: str = ""
    draft_source: str = "template"
    draft_model: str | None = None


class TriageClient:
    """Thin wrapper over the mock API."""

    def __init__(self, base_url: str, read_token: str, write_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.read_token = read_token
        self.write_token = write_token
        self._client = httpx.Client(timeout=HTTP_TIMEOUT)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TriageClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def get_inbox(self) -> list[dict]:
        response = self._client.get(
            f"{self.base_url}/inbox",
            headers=self._headers(self.read_token),
        )
        response.raise_for_status()
        return response.json()

    def _require_write_token(self) -> str:
        if not self.write_token:
            raise RuntimeError("Write scope required but no write token is configured")
        return self.write_token

    def send_reply(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        in_reply_to: str | None = None,
    ) -> dict:
        token = self._require_write_token()
        payload: dict = {"to": to, "subject": subject, "body": body}
        if in_reply_to is not None:
            payload["in_reply_to"] = in_reply_to
        response = self._client.post(
            f"{self.base_url}/mail/send",
            headers=self._headers(token),
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def send_alert(self, *, channel: str, message: str) -> dict:
        token = self._require_write_token()
        response = self._client.post(
            f"{self.base_url}/slack/alert",
            headers=self._headers(token),
            json={"channel": channel, "message": message},
        )
        response.raise_for_status()
        return response.json()

    def create_lead(
        self,
        *,
        name: str,
        email: str,
        company: str | None = None,
        summary: str | None = None,
    ) -> dict:
        token = self._require_write_token()
        payload: dict = {"name": name, "email": email}
        if company is not None:
            payload["company"] = company
        if summary is not None:
            payload["summary"] = summary
        response = self._client.post(
            f"{self.base_url}/crm/lead",
            headers=self._headers(token),
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def _email_field(email: dict, field: str) -> str:
    value = email.get(field)
    if not value:
        raise ValueError(f"missing required email field: {field}")
    return value


def _sender_name(from_addr: str) -> str:
    if "@" in from_addr:
        return from_addr.split("@", 1)[0]
    return from_addr


# Multi-label public suffixes we care about, so the registrable domain is taken
# as the label *before* the suffix (e.g. acme.co.uk -> "acme", not "co").
_COMPOUND_SUFFIXES = ("co.uk", "com.au", "co.nz", "co.jp", "com.br", "co.in")

_FREE_PROVIDERS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "icloud.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
}


def _company_from_email(from_addr: str) -> str | None:
    """Best-effort company name from the sender's email domain.

    Keys off the registrable domain (the label just before the public suffix),
    so subdomains collapse consistently: mail.acme.com and eu.acme.com both
    yield "Acme". This is a heuristic, not identity resolution — it cannot
    canonicalize a company that uses unrelated domains (acme.com vs acme.io);
    that belongs to CRM-side dedup or the human approver.

    Skips public/free mail providers and returns None when no useful inference
    is possible.
    """
    if "@" not in from_addr:
        return None
    domain = from_addr.split("@", 1)[1].strip().lower().rstrip(".")
    if not domain or "." not in domain:
        return None
    if domain in _FREE_PROVIDERS:
        return None

    labels = domain.split(".")
    # Strip a known compound suffix (co.uk) or a single TLD (.com) to find the
    # registrable label, regardless of how many subdomains precede it.
    if len(labels) >= 3 and ".".join(labels[-2:]) in _COMPOUND_SUFFIXES:
        registrable = labels[-3]
    else:
        registrable = labels[-2]

    return registrable.replace("-", " ").title() or None


def _reply_body(label: str) -> str:
    if label not in REPLY_TEMPLATES:
        raise ValueError(f"No reply template for label: {label}")
    return REPLY_TEMPLATES[label]


def plan_actions(label: str, email: dict) -> list[ProposedAction]:
    """Turn a classification into the actions it implies, per the routing table."""
    if label not in ROUTING:
        return []

    sender = _email_field(email, "from")
    subject = _email_field(email, "subject")
    email_id = _email_field(email, "id")

    actions: list[ProposedAction] = []
    for kind in ROUTING[label]:
        if kind == "send_reply":
            actions.append(
                ProposedAction(
                    kind="send_reply",
                    payload={
                        "to": sender,
                        "subject": f"Re: {subject}",
                        "body": _reply_body(label),
                        "in_reply_to": email_id,
                    },
                    rationale=f"Draft reply to customer ({label})",
                    draft_source="template",
                )
            )
        elif kind == "send_alert":
            body = email.get("body", "")
            message = f"Bug report: {subject}\n\n{body[:500]}"
            actions.append(
                ProposedAction(
                    kind="send_alert",
                    payload={"channel": "#engineering", "message": message},
                    rationale="Alert engineering about product bug",
                )
            )
        elif kind == "create_lead":
            payload: dict = {
                "name": _sender_name(sender),
                "email": sender,
                "summary": subject,
            }
            company = _company_from_email(sender)
            if company is not None:
                payload["company"] = company
            actions.append(
                ProposedAction(
                    kind="create_lead",
                    payload=payload,
                    rationale="Create CRM lead for sales inquiry",
                )
            )
    return actions
