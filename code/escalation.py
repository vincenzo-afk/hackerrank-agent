from __future__ import annotations

import re
import os
from typing import Pattern

from utils import debug_log

# ---------------------------------------------------------------------------
# Patterns — loaded from env overrides or defaults, compiled once at import
# ---------------------------------------------------------------------------

def _env_list(var: str, default: list[str]) -> list[Pattern]:
    """Allow env var to extend default patterns; returns compiled regexes."""
    extra = os.getenv(var, "")
    patterns = default + [p.strip() for p in extra.split(";") if p.strip()]
    return [re.compile(pat, flags=re.IGNORECASE | re.DOTALL) for pat in patterns]


_INJECTION_PATTERNS = _env_list("ESCALATION_INJECTION_EXTRA", [
    r"\b(system prompt|developer message|internal rules|exact logic|retrieved documents)\b",
    r"\b(show|reveal|display)\b.*\b(prompt|rules|documents|logic)\b",
    r"\b(ignore (all|previous) instructions)\b",
    r"\bprint\b.*\b(system|prompt)\b",
    r"\b(règles internes|documents récupérés|logique exacte|affiche)\b",
    r"\b(act as|pretend to be|roleplay as)\b.*\b(admin|root|developer|god mode)\b",
    r"\b(jailbreak|dan mode|do anything now)\b",
])

_MALICIOUS_PATTERNS = _env_list("ESCALATION_MALICIOUS_EXTRA", [
    r"\bdelete all files\b",
    r"\b(format|wipe)\b.*\b(disk|drive)\b",
    r"\brm\s+-rf\b",
    r"\bshutdown\b",
    r"\bexploit\b.*\b(vulnerability|system)\b",
    r"\b(drop table|truncate table|delete from)\b",
    r"\b(exec|eval|os\.system|subprocess)\b.*\(",
])

_SECURITY_REPORT_PATTERNS = _env_list("ESCALATION_SECURITY_EXTRA", [
    r"\b(security vulnerability|major vulnerability|found a vulnerability|bug bounty|security issue)\b",
])

_IDENTITY_THEFT_PATTERNS = _env_list("ESCALATION_IDENTITY_EXTRA", [
    r"\b(identity theft|my identity has been stolen|my identity was stolen)\b",
])

_FRAUD_PATTERNS = _env_list("ESCALATION_FRAUD_EXTRA", [
    r"\b(fraud|fraudulent|stolen card|lost card|card was stolen|card has been stolen|unauthorized transaction)\b",
])

_NON_OWNER_ACCESS_PATTERNS = _env_list("ESCALATION_NON_OWNER_EXTRA", [
    r"\b(i am not (the )?(owner|admin)|not an admin|not the account owner)\b",
    r"\bplease restore (my )?access\b",
])

_SCORE_MANIPULATION_PATTERNS = _env_list("ESCALATION_SCORE_EXTRA", [
    r"\b(change my score|update my score|fix my score|score dispute|increase my score)\b",
    r"\b(reverse (the )?(decision|result))\b",
    r"\brecruiter\b.*\b(reverse|change|rejected|move me)\b",
    r"\b(graded me unfairly|must have graded me)\b",
])

_OUTAGE_PATTERNS = _env_list("ESCALATION_OUTAGE_EXTRA", [
    r"\b(stopped working completely|all requests are failing|site is down|service is down|outage)\b",
    r"\b(none of the submissions.*working|submissions.*not working across)\b",
])

_SENSITIVE_TOPICS = _env_list("ESCALATION_SENSITIVE_EXTRA", [
    r"\b(billing|refund|chargeback|payment dispute|dispute)\b",
    r"\b(account deletion|delete my account|close my account)\b",
    r"\b(legal|lawyer|gdpr|compliance|sue you|legal action|attorney)\b",
])

_PII_PATTERNS = _env_list("ESCALATION_PII_EXTRA", [
    r"\b(ssn|social security number|passport|national insurance|dob|date of birth)\b",
    r"\b\d{3}-\d{2}-\d{4}\b",         # US SSN pattern
    r"\b\d{16}\b",                     # Card number (16 digits)
])

_HARM_DISCRIMINATION_PATTERNS = _env_list("ESCALATION_HARM_EXTRA", [
    r"\b(suicide|kill myself|harm myself|end my life)\b",
    r"\b(racist|sexist|homophobic|discriminate|discrimination|harassment|hate speech)\b",
])

_URGENT_CASH_PATTERNS = _env_list("ESCALATION_CASH_EXTRA", [
    r"\burgent cash\b",
    r"\bneed cash\b",
    r"\bonly the visa card\b",
    r"\bneed.*cash.*right now\b",
])

_DEMAND_ACTION_PATTERNS = _env_list("ESCALATION_DEMAND_EXTRA", [
    r"\b(refund me|ban the seller|block the merchant|make visa refund|chargeback|reverse the charge)\b",
])

_VISA_FAQ_PATTERNS = _env_list("ESCALATION_VISA_FAQ_EXTRA", [
    r"\bhow do i (dispute a charge|log in to my account|find an atm)\b",
    r"\bwhat should i do if my visa card has been lost or stolen\b",
    r"\bwhere can i report a lost or stolen visa card\b",
    r"\bwhy was my card declined\b",
])

_NON_ENGLISH_HINTS = [
    re.compile(r"[àâçéèêëîïôùûüÿœæ]", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(bonjour|merci|svp|s'il vous plaît|je veux|affiche)\b", flags=re.IGNORECASE | re.DOTALL),
]

_REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{8,}")  # 9+ same chars in a row (spam)


def _matches_any(patterns: list[Pattern], text: str) -> bool:
    for pat in patterns:
        if pat.search(text):
            return True
    return False


class EscalationEngine:
    def should_escalate(self, issue: str, company: str | None, chunks: list[dict], is_pre_retrieval: bool = False) -> tuple[bool, str]:
        text = f"{issue or ''}".strip()
        lc = text.lower()

        # ── Spam / garbage input ──────────────────────────────────────────
        if _REPEATED_CHAR_PATTERN.search(text):
            return True, "spam: repeated characters / garbage input"

        # ── Prompt injection ──────────────────────────────────────────────
        if _matches_any(_INJECTION_PATTERNS, text):
            if _matches_any(_NON_ENGLISH_HINTS, text):
                res = (True, "prompt injection attempt in non-English requesting internal rules/docs")
            else:
                res = (True, "prompt injection attempt requesting internal rules/docs")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "run"), hypothesis_id="H3",
                      location="escalation.py:should_escalate", message="Escalate: injection", data={"reason": res[1]})
            return res

        # ── Malicious / destructive commands ─────────────────────────────
        if _matches_any(_MALICIOUS_PATTERNS, text):
            res = (True, "malicious: destructive/system action request")
            debug_log(run_id=os.getenv("DEBUG_RUN_ID", "run"), hypothesis_id="H3",
                      location="escalation.py:should_escalate", message="Escalate: malicious", data={"reason": res[1]})
            return res

        # ── PII detected ──────────────────────────────────────────────────
        if _matches_any(_PII_PATTERNS, text):
            return True, "privacy: contains highly sensitive PII"

        # ── Self-harm / discrimination ────────────────────────────────────
        if _matches_any(_HARM_DISCRIMINATION_PATTERNS, text):
            return True, "safety: self-harm, harassment, or discrimination reported"

        # ── Identity theft ────────────────────────────────────────────────
        if _matches_any(_IDENTITY_THEFT_PATTERNS, text):
            return True, "sensitive_financial: identity theft report requires human review"

        # ── Fraud / stolen card ───────────────────────────────────────────
        if _matches_any(_FRAUD_PATTERNS, text):
            return True, "sensitive_financial: fraud/lost-stolen-card requires human review"

        # ── Non-owner requesting access ───────────────────────────────────
        if _matches_any(_NON_OWNER_ACCESS_PATTERNS, text):
            return True, "auth_boundary: access requested by non-owner/admin"

        # ── Score manipulation ────────────────────────────────────────────
        if _matches_any(_SCORE_MANIPULATION_PATTERNS, text):
            return True, "policy: score/decision change request is not allowed"

        # ── Service outage ────────────────────────────────────────────────
        if _matches_any(_OUTAGE_PATTERNS, text):
            return True, "outage: possible widespread outage; needs human triage"

        # ── Urgent cash / card-only ───────────────────────────────────────
        if _matches_any(_URGENT_CASH_PATTERNS, text):
            return True, "sensitive_financial: urgent cash request needs human review"

        # ── Aggressive financial demands ──────────────────────────────────
        if _matches_any(_DEMAND_ACTION_PATTERNS, text):
            return True, "sensitive_financial: user demands refund/chargeback requiring human review"

        # ── Known Visa FAQ → don't escalate ──────────────────────────────
        if company == "visa" and _matches_any(_VISA_FAQ_PATTERNS, text):
            return False, ""

        # ── Security / bug bounty ─────────────────────────────────────────
        if _matches_any(_SECURITY_REPORT_PATTERNS, text):
            if chunks:
                blob = "\n".join([(c.get("text") or "") for c in chunks]).lower()
                if any(k in blob for k in ["bug bounty", "security report", "responsible disclosure", "security@", "vulnerability"]):
                    return False, ""
                return True, "security: vulnerability report not covered by documentation"
            # Defer to post-retrieval if we haven't retrieved yet
            if not is_pre_retrieval:
                return True, "security: vulnerability report not covered by documentation"

        # ── The following checks rely on the ABSENCE of documentation ──────
        if not is_pre_retrieval:
            # ── Out-of-scope / no context ─────────────────────────────────────
            if company is None and not chunks:
                if not any(w in lc for w in ["account", "billing", "payment", "error", "access", "help", "support", "issue", "card", "api", "test", "assessment"]):
                    return True, "out_of_scope: no relevant documentation found"

            # ── Vague with no docs ────────────────────────────────────────────
            if company is None and not chunks:
                if len(lc) < 30 or lc in {"it's not working", "its not working", "not working", "help"}:
                    return True, "vague: insufficient details and no documentation match"

            # ── Sensitive topic with no supporting docs ───────────────────────
            if not chunks and _matches_any(_SENSITIVE_TOPICS, text):
                return True, "sensitive: topic with no supporting documentation found"

        return False, ""
