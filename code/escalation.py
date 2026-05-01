from __future__ import annotations

import re

import os

from utils import debug_log

_INJECTION_PATTERNS = [
    r"\b(system prompt|developer message|internal rules|exact logic|retrieved documents)\b",
    r"\b(show|reveal|display)\b.*\b(prompt|rules|documents|logic)\b",
    r"\b(ignore (all|previous) instructions)\b",
    r"\bprint\b.*\b(system|prompt)\b",
    # French injection-ish
    r"\b(règles internes|documents récupérés|logique exacte|affiche)\b",
]

_MALICIOUS_PATTERNS = [
    r"\bdelete all files\b",
    r"\b(format|wipe)\b.*\b(disk|drive)\b",
    r"\brm\s+-rf\b",
    r"\bshutdown\b",
    r"\bexploit\b.*\b(vulnerability|system)\b",
]

_SECURITY_REPORT_PATTERNS = [
    r"\b(security vulnerability|major vulnerability|found a vulnerability|bug bounty|security issue)\b",
]

_IDENTITY_THEFT_PATTERNS = [
    r"\b(identity theft|my identity has been stolen|my identity was stolen)\b",
]

_FRAUD_PATTERNS = [
    r"\b(fraud|fraudulent|stolen card|lost card|card was stolen|card has been stolen|unauthorized transaction)\b",
]

_NON_OWNER_ACCESS_PATTERNS = [
    r"\b(i am not (the )?(owner|admin)|not an admin|not the account owner)\b",
    r"\bplease restore (my )?access\b",
]

_SCORE_MANIPULATION_PATTERNS = [
    r"\b(change my score|update my score|fix my score|score dispute|increase my score)\b",
    r"\b(reverse (the )?(decision|result))\b",
    r"\brecruiter\b.*\b(reverse|change|rejected|move me)\b",
    r"\b(graded me unfairly|must have graded me)\b",
]

_OUTAGE_PATTERNS = [
    r"\b(stopped working completely|all requests are failing|site is down|service is down|outage)\b",
    r"\b(none of the submissions.*working|submissions.*not working across)\b",
]

_SENSITIVE_TOPICS = [
    r"\b(billing|refund|chargeback|payment dispute|dispute)\b",
    r"\b(account deletion|delete my account|close my account)\b",
    r"\b(legal|lawyer|gdpr|compliance)\b",
]

_URGENT_CASH_PATTERNS = [
    r"\burgent cash\b",
    r"\bneed cash\b",
    r"\bonly the visa card\b",
    r"\bneed.*cash.*right now\b",
]

_DEMAND_ACTION_PATTERNS = [
    r"\b(refund me|ban the seller|block the merchant|make visa refund|chargeback|reverse the charge)\b",
]

_VISA_FAQ_PATTERNS = [
    r"\bhow do i (dispute a charge|log in to my account|find an atm)\b",
    r"\bwhat should i do if my visa card has been lost or stolen\b",
    r"\bwhere can i report a lost or stolen visa card\b",
    r"\bwhy was my card declined\b",
]

_NON_ENGLISH_HINTS = [
    r"[àâçéèêëîïôùûüÿœæ]",
    r"\b(bonjour|merci|svp|s'il vous plaît|je veux|affiche)\b",
]


def _matches_any(patterns: list[str], text: str) -> bool:
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            return True
    return False


class EscalationEngine:
    def should_escalate(self, issue: str, company: str | None, chunks: list[dict]) -> tuple[bool, str]:
        text = f"{issue or ''}".strip()
        lc = text.lower()

        # 5.1 hard pre-checks
        if _matches_any(_INJECTION_PATTERNS, text):
            # Non-English injection is explicitly called out in plan
            if _matches_any(_NON_ENGLISH_HINTS, text):
                res = (True, "prompt injection attempt in non-English requesting internal rules/docs")
                debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: injection non-English", data={"company": company, "reason": res[1]})
                return res
            res = (True, "prompt injection attempt requesting internal rules/docs")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: injection", data={"company": company, "reason": res[1]})
            return res

        if _matches_any(_MALICIOUS_PATTERNS, text):
            res = (True, "malicious: destructive/system action request")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: malicious", data={"company": company, "reason": res[1]})
            return res

        if _matches_any(_IDENTITY_THEFT_PATTERNS, text):
            res = (True, "sensitive_financial: identity theft report requires human review")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: identity theft", data={"company": company, "reason": res[1]})
            return res

        if _matches_any(_FRAUD_PATTERNS, text):
            res = (True, "sensitive_financial: fraud/lost-stolen-card related issue requires human review")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: fraud/lost", data={"company": company, "reason": res[1]})
            return res

        if _matches_any(_NON_OWNER_ACCESS_PATTERNS, text):
            res = (True, "auth_boundary: access requested by non-owner/admin")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: non-owner access", data={"company": company, "reason": res[1]})
            return res

        if _matches_any(_SCORE_MANIPULATION_PATTERNS, text):
            res = (True, "policy: score/decision change request is not allowed")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: score manipulation", data={"company": company, "reason": res[1]})
            return res

        if _matches_any(_OUTAGE_PATTERNS, text):
            res = (True, "outage: possible widespread outage; needs human triage")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: outage", data={"company": company, "reason": res[1]})
            return res

        # Urgent-cash / card-only = high-risk financial (Plan.md Category G)
        if _matches_any(_URGENT_CASH_PATTERNS, text):
            res = (True, "sensitive_financial: urgent cash request needs human review")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: urgent cash", data={"company": company, "reason": res[1]})
            return res

        # Aggressive demands for refunds/bans/chargebacks from Visa need human review
        if _matches_any(_DEMAND_ACTION_PATTERNS, text):
            res = (True, "sensitive_financial: user demands refund, chargeback, or merchant action requiring human review")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: demand action", data={"company": company, "reason": res[1]})
            return res

        # Known Visa FAQ questions should be answered from corpus, not escalated
        if company == "visa" and _matches_any(_VISA_FAQ_PATTERNS, text):
            return False, ""

        # Security / bug bounty reports: prefer routing; only allow reply if corpus covers it.
        # When called pre-retrieval (chunks empty) we defer the decision so the agent can
        # retrieve docs first and then re-evaluate in the post-retrieval check.
        if _matches_any(_SECURITY_REPORT_PATTERNS, text):
            if chunks:
                blob = "\n".join([(c.get("text") or "") for c in chunks]).lower()
                if any(k in blob for k in ["bug bounty", "security report", "responsible disclosure", "security@", "vulnerability"]):
                    return False, ""
                res = (True, "security: vulnerability report not covered by documentation")
                debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H3", location="escalation.py:should_escalate", message="Escalate: security uncovered", data={"company": company, "reason": res[1]})
                return res
            # Defer until post-retrieval when we have chunks to inspect.

        # 5.1 out-of-scope with no corpus match (only when company is unknown)
        if company is None and (not chunks):
            # If it doesn't look like a support issue at all, escalate as OOS.
            if not any(w in lc for w in ["account", "billing", "payment", "error", "access", "help", "support", "issue"]):
                res = (True, "out_of_scope: no relevant documentation found")
                debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H4", location="escalation.py:should_escalate", message="Escalate: out of scope", data={"company": company, "reason": res[1]})
                return res

        # Vague + no company + no docs (plan prefers escalate for safety)
        if company is None and not chunks:
            if len(lc) < 30 or lc in {"it's not working", "its not working", "not working", "help"}:
                res = (True, "vague: insufficient details and no documentation match")
                debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H4", location="escalation.py:should_escalate", message="Escalate: vague", data={"company": company, "reason": res[1]})
                return res

        # 5.2 corpus-informed escalation after retrieval
        if not chunks and _matches_any(_SENSITIVE_TOPICS, text):
            res = (True, "sensitive: topic with no supporting documentation found")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"), hypothesis_id="H4", location="escalation.py:should_escalate", message="Escalate: sensitive no docs", data={"company": company, "reason": res[1]})
            return res

        return False, ""

