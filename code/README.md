# Multi-Domain Support Triage Agent (HackerRank Orchestrate)

## Overview
This project implements a local RAG-based triage agent that reads `support_tickets/support_tickets.csv` and writes `output.csv` with:
`status, product_area, response, justification, request_type`.

The agent supports three corpora (HackerRank, Claude, Visa) from the local `data/` folder only (no live web calls).

## Architecture
`main.py` loads tickets, builds a local embedding index over `data/`, retrieves the most relevant chunks per ticket, applies escalation rules, classifies product area/request type, and (when safe) calls Groq to generate a response grounded in retrieved documentation.

## Setup
```bash
cd code/
pip install -r requirements.txt
cp ../.env.example ../.env
# Add GROQ_API_KEY to .env
```

## Run
```bash
python main.py
# Output written to output.csv
```

## Design decisions
- Local embeddings (`sentence-transformers`) keep retrieval offline and deterministic.
- Rule-based escalation runs before any LLM call to block prompt-injection, malicious commands, and sensitive/high-risk topics.
- Responses are generated with temperature \(0\) and must be grounded in retrieved docs; if docs are missing, the agent escalates instead of guessing.

## Known limitations
- Product area classification is heuristic (retrieval + keyword mapping), not a trained classifier.
- If the local corpus is incomplete for a topic, tickets will be escalated more often to avoid hallucinations.
- Non-English handling is intentionally conservative (suspicious internal-logic requests escalate).

