from __future__ import annotations

import re


def infer_company(issue: str, subject: str, company: str | None, retrieved_chunks: list[dict]) -> str | None:
    if company is not None:
        return company

    text = f"{subject or ''}\n{issue or ''}".lower()
    if any(k in text for k in ["hackerrank", "assessment", "assessments", "test", "tests", "candidate", "recruiter", "interview"]):
        return "hackerrank"
    if any(k in text for k in ["claude", "anthropic", "conversation", "workspace", "model", "api key"]):
        return "claude"
    if any(k in text for k in ["visa", "card", "transaction", "merchant", "charge", "cheque"]):
        return "visa"

    # Heuristic: pick best-matching company from top chunk
    if retrieved_chunks:
        return str(retrieved_chunks[0].get("source_company") or "generic").lower()

    return None


def classify_product_area(company: str | None, issue: str, subject: str, retrieved_chunks: list[dict]) -> str:
    if company is None:
        return "out_of_scope" if not retrieved_chunks else "general_support"

    c = company.lower()
    text = f"{subject or ''}\n{issue or ''}".lower()
    top_file = (retrieved_chunks[0].get("filename") if retrieved_chunks else "") or ""
    hint = f"{top_file} {text}".lower()

    if c == "hackerrank":
        if any(k in hint for k in ["billing", "invoice", "subscription", "payment", "refund"]):
            return "billing"
        if any(k in hint for k in ["api", "integration", "webhook", "sso", "saml", "lti"]):
            return "api_integration"
        if any(k in hint for k in ["candidate", "invitation", "invite", "resume", "application"]):
            return "candidates"
        if any(k in hint for k in ["assessment", "test", "challenge", "submission"]):
            return "assessments"
        if "interview" in hint:
            return "interview"
        if any(k in hint for k in ["account", "login", "password", "mfa", "2fa", "access"]):
            return "account"
        if "screen" in hint:
            return "screen"
        if "community" in hint:
            return "community"
        return "general_support"

    if c == "claude":
        if any(k in hint for k in ["security", "vulnerability", "bug bounty", "exploit"]):
            return "security"
        if any(k in hint for k in ["billing", "invoice", "subscription", "payment", "refund"]):
            return "billing"
        if any(k in hint for k in ["privacy", "data", "gdpr", "retention", "delete"]):
            return "privacy"
        if any(k in hint for k in ["api", "integration", "bedrock", "aws", "key", "lti", "sso", "saml"]):
            return "api_integration"
        if any(k in hint for k in ["workspace", "admin", "member", "role", "access"]):
            return "workspace_admin"
        if any(k in hint for k in ["conversation", "chat", "thread", "history", "export"]):
            return "conversation_management"
        if any(k in hint for k in ["hallucination", "refuse", "policy", "safety", "behavior"]):
            return "model_behavior"
        return "general_support"

    if c == "visa":
        if any(k in hint for k in ["fraud", "identity", "stolen", "lost", "unauthorized"]):
            return "fraud_security"
        if any(k in hint for k in ["dispute", "chargeback", "refund", "reversal"]):
            return "dispute_resolution"
        if any(k in hint for k in ["travel", "abroad", "international", "trip"]):
            return "travel_support"
        if any(k in hint for k in ["merchant", "terminal", "pos"]):
            return "merchant_support"
        if any(k in hint for k in ["card", "pin", "replacement", "limit", "block", "unblock"]):
            return "card_management"
        return "general_support"

    return "general_support"


_TRIVIAL_PATTERNS = [
    "thank you for helping me",
    "hi there",
    "hello",
    "thanks",
    "thank you",
]


def classify_request_type(issue: str, subject: str) -> str:
    text = f"{subject or ''}\n{issue or ''}".lower()

    if any(k in text for k in ["delete all files", "rm -rf", "format disk", "wipe drive"]):
        return "invalid"

    # Trivial / non-actionable greetings
    stripped = " ".join(text.split())
    if any(stripped == t or stripped.startswith(t + " ") for t in _TRIVIAL_PATTERNS):
        return "invalid"

    # Infosec / form-filling / LTI setup / hiring-process requests are feature-like scope asks
    if any(k in text for k in ["infosec", "fill in the forms", "setup a claude lti key", "claude lti key", "infosec forms", "hiring process", "onboarding forms"]):
        return "feature_request"

    if any(k in text for k in ["feature request", "can you add", "would be great if", "suggest", "request a feature", "add support for", "implement", "extend"]):
        return "feature_request"

    if any(k in text for k in ["bug", "glitch", "unexpected", "crash", "crashes", "site is down", "is down", "pages are inaccessible"]):
        return "bug"

    if any(k in text for k in ["not working", "can't", "cannot", "stopped", "error", "broken", "lost access", "failing"]):
        return "product_issue"

    # Default to product_issue for typical support triage (plan.md prioritizes routing),
    # reserving "invalid" for clearly malicious / non-actionable prompts.
    return "product_issue"

