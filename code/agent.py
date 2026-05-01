from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env FIRST so all os.getenv calls below pick up the values
load_dotenv()

from groq import Groq

from classifier import classify_product_area, classify_request_type, infer_company
from escalation import EscalationEngine
from prompts import ESCALATION_MESSAGE_TEMPLATE, RESPONSE_PROMPT_TEMPLATE, SYSTEM_PROMPT
from retriever import CorpusRetriever
from utils import debug_log, normalize_company, sanitize_plaintext

# Configurable via .env — read after load_dotenv()
_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", 3))
_RETRY_BACKOFF = float(os.getenv("GROQ_RETRY_BACKOFF", 5.0))
_RATE_LIMIT_WAIT_CAP = float(os.getenv("GROQ_RATE_LIMIT_WAIT_CAP", 120.0))

# Regex to parse "try again in 2m3.5s" from Groq 429 messages
_RETRY_AFTER_RE = re.compile(r"try again in (\d+)m([\d.]+)s", re.IGNORECASE)


@dataclass(frozen=True)
class TicketResult:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str


class TriageAgent:
    def __init__(
        self,
        retriever: CorpusRetriever,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> None:
        self.retriever = retriever
        self.escalation = EscalationEngine()

        # load_dotenv() was already called at module level
        self.model = os.getenv("GROQ_MODEL", model)
        self.temperature = float(os.getenv("GROQ_TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("GROQ_MAX_TOKENS", str(max_tokens)))

        self._api_key = os.getenv("GROQ_API_KEY")
        if not self._api_key:
            print("[agent] WARNING: GROQ_API_KEY not set. LLM calls will fail; "
                  "all tickets will be escalated. Set it in your .env file.")
        self._client = Groq(api_key=self._api_key) if self._api_key else None

        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "run"),
            hypothesis_id="H5",
            location="agent.py:__init__",
            message="LLM client initialized",
            data={
                "has_groq_key": bool(self._api_key),
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "max_retries": _MAX_RETRIES,
            },
        )

    def _format_docs(self, chunks: list[dict]) -> str:
        if not chunks:
            return "(No relevant documentation found.)"
        parts: list[str] = []
        for i, ch in enumerate(chunks, start=1):
            src = ch.get("filename", "unknown")
            parts.append(f"--- DOC {i} (source: {src}) ---\n{ch.get('text', '').strip()}")
        return "\n\n".join(parts)

    def _extract_phone_numbers(self, text: str) -> list[str]:
        """Surface phone numbers found in retrieved corpus docs."""
        found = re.findall(r"(?:\+?\d[\d\-\s\(\)]{7,}\d)", text)
        seen: set[str] = set()
        cleaned: list[str] = []
        for f in found:
            s = " ".join(f.split())
            if s not in seen:
                seen.add(s)
                cleaned.append(s)
        return cleaned[:3]

    def _build_justification(
        self,
        status: str,
        product_area: str,
        request_type: str,
        chunks: list[dict],
        escalation_reason: str = "",
    ) -> str:
        if status == "escalated":
            return f"Escalated because: {escalation_reason}. Human review required."

        if chunks:
            seen: set[str] = set()
            sources: list[str] = []
            for c in chunks:
                fn = c.get("filename", "")
                if fn and fn not in seen:
                    seen.add(fn)
                    sources.append(fn)
            source_str = ", ".join(sources[:3])
            return (
                f"Responded based on corpus sources: {source_str}. "
                f"Classified as {request_type} under {product_area}."
            )

        return f"Responded without corpus match. Classified as {request_type} under {product_area}."

    def _escalation_response(
        self,
        company: str | None,
        reason: str,
        product_area: str,
        request_type: str,
        chunks: list[dict] | None = None,
    ) -> TicketResult:
        # BUG FIX: "company or 'the'" produced "contact the support" — use proper fallback
        company_label = company.title() if company else "our"
        chunks = chunks or []
        lc_reason = (reason or "").lower()

        if lc_reason.startswith("malicious:"):
            response = (
                "I can't assist with requests to damage systems or delete files. "
                "If you have a legitimate question, please reach out to support."
            )
            request_type = "invalid"

        elif lc_reason.startswith("spam:"):
            response = (
                "Your message couldn't be processed. Please send a clear, specific support question."
            )
            request_type = "invalid"

        elif lc_reason.startswith("privacy:"):
            response = (
                "Please do not share sensitive personal information such as SSNs, "
                "passport numbers, or card numbers in chat. For security-sensitive issues, "
                "please contact support directly through official channels."
            )

        elif lc_reason.startswith("safety:"):
            response = (
                "We've escalated your message to a human specialist who can help. "
                "If this is an emergency, please contact the relevant emergency services."
            )

        elif lc_reason.startswith("policy:"):
            response = (
                "This request requires review by a human support specialist. "
                "We can't change test scores or reverse hiring decisions via automated support."
            )

        elif lc_reason.startswith("auth_boundary:"):
            response = (
                "This request requires review by a human support specialist. "
                "Please contact your workspace admin or official support to restore access."
            )

        elif lc_reason.startswith("outage:"):
            response = (
                "This may relate to a broader service outage. A human specialist has been notified. "
                "Please check the status page for live updates."
            )

        elif lc_reason.startswith("out_of_scope:"):
            response = (
                "This request is outside the scope of this support agent. "
                "Please contact the appropriate support team for assistance."
            )
            if request_type == "product_issue":
                request_type = "invalid"

        elif lc_reason.startswith("vague:"):
            response = (
                "There isn't enough detail to assist safely. Please provide more information "
                "such as the exact error message and steps to reproduce the issue."
            )
            request_type = "invalid"

        elif lc_reason.startswith("sensitive_financial:"):
            blob = "\n".join([(c.get("text") or "") for c in chunks])
            phones = self._extract_phone_numbers(blob)
            if phones:
                response = (
                    f"This request requires review by a {company_label} support specialist. "
                    "Please contact support directly. "
                    "Phone numbers from the documentation: " + "; ".join(phones) + "."
                )
            else:
                response = ESCALATION_MESSAGE_TEMPLATE.format(company=company_label)

        elif lc_reason.startswith("security:"):
            response = (
                "This request requires review by a human security specialist. "
                "Please follow the documented responsible disclosure or bug bounty process."
            )

        elif lc_reason.startswith("prompt injection"):
            response = (
                "I can't process that request. If you have a genuine support question, "
                "please ask it directly."
            )
            request_type = "invalid"

        else:
            response = ESCALATION_MESSAGE_TEMPLATE.format(company=company_label)

        justification = self._build_justification(
            status="escalated",
            product_area=product_area,
            request_type=request_type,
            chunks=chunks,
            escalation_reason=reason,
        )

        return TicketResult(
            status="escalated",
            product_area=product_area,
            response=response.strip(),
            justification=justification.strip(),
            request_type=request_type,
        )

    def classify(
        self, company: str | None, issue: str, subject: str, chunks: list[dict]
    ) -> tuple[str, str]:
        product_area = classify_product_area(company, issue, subject, chunks)
        request_type = classify_request_type(issue, subject)
        return product_area, request_type

    def generate_response(
        self,
        company: str | None,
        issue: str,
        subject: str,
        chunks: list[dict],
        conversation_history: list[dict] | None = None,
    ) -> str:
        if not self._client:
            return ""
        if not issue or not issue.strip():
            return ""

        docs_block = self._format_docs(chunks)
        user_msg = RESPONSE_PROMPT_TEMPLATE.format(
            company=company or "unknown",
            issue=issue.strip(),
            docs_block=docs_block,
        )

        # Build messages — system prompt + prior turns + current question
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            max_turns = int(os.getenv("CHAT_HISTORY_TURNS", "6"))
            for h in conversation_history[-max_turns * 2:]:   # *2: each turn = user+assistant
                role = h.get("role", "")
                content = h.get("content", "")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_msg})

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=messages,
                )
                content = None
                try:
                    content = completion.choices[0].message.content
                except (IndexError, AttributeError):
                    content = None
                return sanitize_plaintext(content or "")

            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                is_rate_limit = (
                    "rate_limit" in err_str.lower()
                    or "429" in err_str
                    or "tokens per day" in err_str.lower()
                )

                if is_rate_limit and attempt < _MAX_RETRIES:
                    m = _RETRY_AFTER_RE.search(err_str)
                    if m:
                        wait = int(m.group(1)) * 60 + float(m.group(2))
                    else:
                        wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                    wait = min(wait, _RATE_LIMIT_WAIT_CAP)

                    debug_log(
                        run_id=os.getenv("DEBUG_RUN_ID", "run"),
                        hypothesis_id="H5",
                        location="agent.py:generate_response",
                        message=f"Rate limit hit (attempt {attempt}/{_MAX_RETRIES}); waiting {wait:.1f}s",
                        data={"attempt": attempt, "wait": wait},
                    )
                    print(f"[agent] Rate limit — waiting {wait:.0f}s before retry "
                          f"({attempt}/{_MAX_RETRIES})…", flush=True)
                    time.sleep(wait)
                else:
                    # Non-retryable OR last attempt — log and give up
                    debug_log(
                        run_id=os.getenv("DEBUG_RUN_ID", "run"),
                        hypothesis_id="H5",
                        location="agent.py:generate_response",
                        message="LLM call failed (non-retryable or retries exhausted)",
                        data={"attempt": attempt, "error": str(exc)[:300]},
                    )
                    break

        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "run"),
            hypothesis_id="H5",
            location="agent.py:generate_response",
            message="All LLM retries exhausted — escalating",
            data={"error": str(last_exc)[:300]},
        )
        return ""  # Caller will escalate on empty response

    def process_ticket(
        self, row, conversation_history: list[dict] | None = None
    ) -> dict:
        # Safely extract and coerce all fields to str
        issue = str(row.get("Issue", "") or "").strip()
        subject = str(row.get("Subject", "") or "").strip()
        company_raw = str(row.get("Company", "") or "").strip()

        # Guard: blank issue cannot be triaged
        if not issue:
            return TicketResult(
                status="escalated",
                product_area="out_of_scope",
                response="No issue text was provided. Please describe your problem.",
                justification="Empty issue field — cannot triage.",
                request_type="invalid",
            ).__dict__

        company = normalize_company(company_raw)
        run_id = os.getenv("DEBUG_RUN_ID", "run")

        # ── Step 1: Pre-retrieval escalation checks ───────────────────────
        pre_escalate, pre_reason = self.escalation.should_escalate(issue, company, chunks=[])
        if pre_escalate:
            product_area, request_type = self.classify(company, issue, subject, [])
            debug_log(run_id=run_id, hypothesis_id="H3",
                      location="agent.py:process_ticket",
                      message="Pre-escalated ticket",
                      data={"company": company, "product_area": product_area, "reason": pre_reason})
            return self._escalation_response(
                company, pre_reason, product_area, request_type, chunks=[]
            ).__dict__

        # ── Step 2: Build retrieval query ─────────────────────────────────
        query_parts = [issue]
        if subject and subject.lower() not in {"help", "help needed", "support", "issue", ""}:
            query_parts.append(subject)
        query = " ".join(query_parts)

        # ── Step 3: Retrieve with company boost ───────────────────────────
        try:
            chunks = self.retriever.retrieve(query=query, company=company, top_k=5)
        except Exception as exc:
            debug_log(run_id=run_id, hypothesis_id="H2",
                      location="agent.py:process_ticket",
                      message="Retrieval failed", data={"error": str(exc)[:200]})
            chunks = []

        # ── Step 4: Infer company if not provided ─────────────────────────
        company = infer_company(issue, subject, company, chunks)
        if company is None:
            # Re-retrieve without company filter across all corpora
            try:
                chunks = self.retriever.retrieve(query=query, company=None, top_k=5)
            except Exception:
                chunks = []
            company = infer_company(issue, subject, None, chunks)

        debug_log(run_id=run_id, hypothesis_id="H2",
                  location="agent.py:process_ticket",
                  message="Company inference completed",
                  data={"company_raw": company_raw, "company_inferred": company, "chunks": len(chunks)})

        # ── Step 5: Post-retrieval escalation check ───────────────────────
        post_escalate, post_reason = self.escalation.should_escalate(issue, company, chunks=chunks)
        if post_escalate:
            product_area, request_type = self.classify(company, issue, subject, chunks)
            debug_log(run_id=run_id, hypothesis_id="H4",
                      location="agent.py:process_ticket",
                      message="Post-retrieval escalated ticket",
                      data={"company": company, "product_area": product_area, "reason": post_reason})
            return self._escalation_response(
                company, post_reason, product_area, request_type, chunks=chunks
            ).__dict__

        # ── Step 6: Classify ──────────────────────────────────────────────
        product_area, request_type = self.classify(company, issue, subject, chunks)

        # ── Step 7: Generate LLM response ────────────────────────────────
        try:
            response = self.generate_response(
                company, issue, subject, chunks,
                conversation_history=conversation_history,
            )
        except Exception as exc:
            debug_log(run_id=run_id, hypothesis_id="H5",
                      location="agent.py:process_ticket",
                      message="generate_response raised unexpectedly",
                      data={"error": str(exc)[:200]})
            response = ""

        if not response:
            debug_log(run_id=run_id, hypothesis_id="H5",
                      location="agent.py:process_ticket",
                      message="LLM returned empty — escalating",
                      data={"company": company, "product_area": product_area})
            return self._escalation_response(
                company, "LLM produced empty response",
                product_area, request_type, chunks=chunks,
            ).__dict__

        # ── Step 8: Build traceable justification ─────────────────────────
        justification = self._build_justification(
            status="replied",
            product_area=product_area,
            request_type=request_type,
            chunks=chunks,
        )

        return TicketResult(
            status="replied",
            product_area=product_area,
            response=sanitize_plaintext(response),
            justification=justification.strip(),
            request_type=request_type,
        ).__dict__