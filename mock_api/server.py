"""Mock client API for the Inbox Triage take-home.

This stands in for a real client's systems: an inbox to read from, an outbound
mail service, and a CRM. You do not need to modify this file (though you may read
it). Run it with `make serve`.

Auth model — read it carefully, the security requirements depend on it:

  * `GET /inbox` accepts EITHER the read token or the write token.
  * All write endpoints (`/mail/send`, `/slack/alert`, `/crm/lead`) require the
    WRITE token specifically. The read token is rejected with 403.

Tokens are read from the environment (see .env.example). Pass them as
`Authorization: Bearer <token>`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

READ_TOKEN = os.environ.get("READ_TOKEN", "read-token-dev")
WRITE_TOKEN = os.environ.get("WRITE_TOKEN", "write-token-dev")

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "emails.json"

app = FastAPI(title="Inbox Triage — Mock Client API", version="1.0.0")

# In-memory record of everything the agent has written, so the grader (and you)
# can inspect side effects. Resets on restart.
_sent_mail: list[dict] = []
_alerts: list[dict] = []
_leads: list[dict] = []


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return authorization.removeprefix("Bearer ").strip()


def require_read(authorization: str | None = Header(default=None)) -> None:
    token = _bearer(authorization)
    if token not in (READ_TOKEN, WRITE_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid token for read scope")


def require_write(authorization: str | None = Header(default=None)) -> None:
    token = _bearer(authorization)
    if token == READ_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Read token cannot perform writes — write scope required",
        )
    if token != WRITE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token for write scope")


class Reply(BaseModel):
    to: str
    subject: str
    body: str
    in_reply_to: str | None = None


class Alert(BaseModel):
    channel: str
    message: str


class Lead(BaseModel):
    name: str
    email: str
    company: str | None = None
    summary: str | None = None


@app.get("/inbox")
def get_inbox(_: None = Depends(require_read)) -> list[dict]:
    return json.loads(FIXTURES.read_text())


@app.post("/mail/send")
def send_mail(reply: Reply, _: None = Depends(require_write)) -> dict:
    record = {"id": f"mail-{len(_sent_mail) + 1}", **reply.model_dump()}
    _sent_mail.append(record)
    return {"status": "sent", **record}


@app.post("/slack/alert")
def send_alert(alert: Alert, _: None = Depends(require_write)) -> dict:
    record = {"id": f"alert-{len(_alerts) + 1}", **alert.model_dump()}
    _alerts.append(record)
    return {"status": "posted", **record}


@app.post("/crm/lead")
def create_lead(lead: Lead, _: None = Depends(require_write)) -> dict:
    record = {"id": f"lead-{len(_leads) + 1}", **lead.model_dump()}
    _leads.append(record)
    return {"status": "created", **record}


@app.get("/_audit")
def audit() -> dict:
    """Inspect side effects produced so far (handy while you build)."""
    return {"sent_mail": _sent_mail, "alerts": _alerts, "leads": _leads}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8099)
