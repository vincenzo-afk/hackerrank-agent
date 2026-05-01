from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from groq import Groq

from classifier import classify_product_area, classify_request_type, infer_company
from escalation import EscalationEngine
from prompts import ESCALATION_MESSAGE_TEMPLATE, RESPONSE_PROMPT_TEMPLATE, SYSTEM_PROMPT
from retriever import CorpusRetriever
from utils import debug_log, normalize_company, sanitize_plaintext


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

        # Allow runtime override without code changes
        self.model = os.getenv("GROQ_MODEL", model)
        self.temperature = temperature
        self.max_tokens = max_tokens

        load_dotenv()
        self._api_key = os.getenv("GROQ_API_KEY")
        self._client = Groq(api_key=self._api_key) if self._api_key else None
        debug_log(
            run_id=os.getenv("DEBUG_RUN_ID", "pre-fix"),
            hypothesis_id="H5",
            location="agent.py:__init__",
            message="LLM client initialized",
            data={"has_groq_key": bool(self._api_key), "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens},
        )

    def _format_docs(self, chunks: list[dict]) -> str:
        if not chunks:
            return "(No relevant documentation found.)"

        parts: list[str] = []
        for i, ch in enumerate(chunks, start=1):
            parts.append(f"--- DOC {i} (source: {ch.get('filename','unknown')}) ---\n{ch.get('text','')}")
        return "\n\n".join(parts)

    def _extract_phone_numbers(self, text: str) -> list[str]:
        # Conservative: only surface numbers present in retrieved docs
        import re

        found = re.findall(r"(?:\+?\d[\d\-\s\(\)]{7,}\d)", text)
        cleaned = []
        for f in found:
            s = " ".join(f.split())
            if s not in cleaned:
                cleaned.append(s)
        return cleaned[:3]

    def _escalation_response(
        self,
        company: str | None,
        reason: str,
        product_area: str,
        request_type: str,
        chunks: list[dict] | None = None,
    ) -> TicketResult:
        company_label = company or "the"
        chunks = chunks or []

        # Special-case messaging per plan.md categories (A–J)
        lc_reason = (reason or "").lower()
        if lc_reason.startswith("malicious:"):
            response = "I can’t help with requests to damage systems or delete files. Please contact support if you have a legitimate, non-destructive request."
            request_type = "invalid"
        elif lc_reason.startswith("policy:"):
            response = "This request requires review by a human support specialist. We can’t change test scores or reverse hiring/recruiter decisions via automated support."
        elif lc_reason.startswith("auth_boundary:"):
            response = "This request requires review by a human support specialist. Please contact your workspace/account admin or official support to restore access."
        elif lc_reason.startswith("outage:"):
            response = "This request requires review by a human support specialist. It may relate to a broader service outage and needs human triage."
        elif lc_reason.startswith("out_of_scope:"):
            response = "This request is out of scope for this support agent and requires human review. Please contact the appropriate support team."
            request_type = "invalid" if request_type == "product_issue" else request_type
        elif lc_reason.startswith("vague:"):
            response = "This request requires human review because there isn’t enough information to troubleshoot safely. Please contact support with details like exact error messages and steps to reproduce."
            request_type = "invalid"
        elif lc_reason.startswith("sensitive_financial:"):
            blob = "\n".join([(c.get("text") or "") for c in chunks])
            phones = self._extract_phone_numbers(blob)
            if phones:
                response = (
                    "This request requires review by a human support specialist due to the sensitive nature of the issue. "
                    f"Please contact {company_label} support directly. Phone numbers found in the documentation: "
                    + "; ".join(phones)
                    + "."
                )
            else:
                response = ESCALATION_MESSAGE_TEMPLATE.format(company=company_label)
        elif lc_reason.startswith("security:"):
            response = "This request requires review by a human support specialist. Please contact the security team or follow the documented responsible disclosure/bug bounty process."
        else:
            response = ESCALATION_MESSAGE_TEMPLATE.format(company=company_label)

        justification = f"Escalated because: {reason}. Human review required."
        return TicketResult(
            status="escalated",
            product_area=product_area,
            response=response.strip(),
            justification=justification.strip(),
            request_type=request_type,
        )

    def classify(self, company: str | None, issue: str, subject: str, chunks: list[dict]) -> tuple[str, str]:
        product_area = classify_product_area(company, issue, subject, chunks)
        request_type = classify_request_type(issue, subject)
        return product_area, request_type

    def generate_response(self, company: str | None, issue: str, subject: str, chunks: list[dict]) -> str:
        if not self._client:
            return ""

        docs_block = self._format_docs(chunks)
        user_msg = RESPONSE_PROMPT_TEMPLATE.format(
            company=company or "unknown",
            subject=subject or "",
            issue=issue or "",
            docs_block=docs_block,
        )

        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )

        content = None
        try:
            content = completion.choices[0].message.content
        except Exception:
            content = None

        return sanitize_plaintext(content or "")

    def process_ticket(self, row) -> dict:
        issue = str(row.get("Issue", "") or "")
        subject = str(row.get("Subject", "") or "")
        company_raw = str(row.get("Company", "") or "")

        company = normalize_company(company_raw)
        run_id = os.getenv("DEBUG_RUN_ID", "pre-fix")

        # Step 2: escalation pre-checks (run BEFORE calling the LLM)
        pre_escalate, pre_reason = self.escalation.should_escalate(issue, company, chunks=[])
        if pre_escalate:
            product_area, request_type = self.classify(company, issue, subject, [])
            debug_log(
                run_id=run_id,
                hypothesis_id="H3",
                location="agent.py:process_ticket",
                message="Pre-escalated ticket",
                data={"company": company, "product_area": product_area, "request_type": request_type, "reason": pre_reason},
            )
            return self._escalation_response(company, pre_reason, product_area, request_type, chunks=[]).__dict__

        # Step 3: infer company if None (may use heuristic retrieval later)
        # Build retrieval query
        query = issue.strip()
        subj = subject.strip()
        if subj and subj.lower() not in {"help", "help needed", "support", "issue"}:
            query = f"{query}\n{subj}".strip()

        # Retrieve with company boost if known
        chunks = self.retriever.retrieve(query=query, company=company, top_k=5)

        inferred = infer_company(issue, subject, company, chunks)
        company = inferred
        if company is None:
            # Re-run retrieval across all corpora; infer from top chunk
            chunks = self.retriever.retrieve(query=query, company=None, top_k=5)
            company = infer_company(issue, subject, None, chunks)
        debug_log(
            run_id=run_id,
            hypothesis_id="H2",
            location="agent.py:process_ticket",
            message="Company inference completed",
            data={"company_raw": company_raw, "company_normalized": normalize_company(company_raw), "company_inferred": company, "chunks": len(chunks)},
        )

        # Step 6: escalation corpus-check
        post_escalate, post_reason = self.escalation.should_escalate(issue, company, chunks=chunks)
        if post_escalate:
            product_area, request_type = self.classify(company, issue, subject, chunks)
            debug_log(
                run_id=run_id,
                hypothesis_id="H4",
                location="agent.py:process_ticket",
                message="Post-retrieval escalated ticket",
                data={"company": company, "product_area": product_area, "request_type": request_type, "reason": post_reason, "chunks": len(chunks)},
            )
            return self._escalation_response(company, post_reason, product_area, request_type, chunks=chunks).__dict__

        # Step 7: classify
        product_area, request_type = self.classify(company, issue, subject, chunks)

        # Step 8: call LLM
        response = self.generate_response(company, issue, subject, chunks)
        if not response:
            # If LLM can't respond, fall back to escalation (avoid empty response)
            debug_log(
                run_id=run_id,
                hypothesis_id="H5",
                location="agent.py:process_ticket",
                message="LLM returned empty response; escalating",
                data={"company": company, "product_area": product_area, "request_type": request_type, "chunks": len(chunks)},
            )
            return self._escalation_response(
                company, "LLM produced empty response", product_area, request_type, chunks=chunks
            ).__dict__

        # Step 9: justification
        source_filename = (chunks[0].get("filename") if chunks else "no_docs")
        justification = f"Responded based on {source_filename}. Issue classified as {request_type} in {product_area}."

        return TicketResult(
            status="replied",
            product_area=product_area,
            response=sanitize_plaintext(response),
            justification=justification.strip(),
            request_type=request_type,
        ).__dict__

