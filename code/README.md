# Multi-Domain Support Triage Agent — code/

## Overview

Terminal-based triage agent that reads `support_tickets/support_tickets.csv` and writes `support_tickets/output.csv` with five output columns:

```
status, product_area, response, justification, request_type
```

Supports three corpora — HackerRank, Claude, Visa — from the local `data/` folder only. No live web calls at inference time.

---

## Architecture

`main.py` orchestrates the full pipeline:

1. **CorpusRetriever** (`retriever.py`) — walks `data/`, strips HTML tags (BeautifulSoup), parses JSON to plain text, chunks documents into ~1200-char overlapping windows, embeds with `all-MiniLM-L6-v2` (sentence-transformers), caches embeddings to `data/embeddings_cache.npy`.
2. **EscalationEngine** (`escalation.py`) — rule-based pre-checks (prompt injection, malicious commands, fraud, identity theft, score manipulation, outages) run *before* any LLM call. Post-retrieval checks catch sensitive topics with no corpus coverage.
3. **Classifier** (`classifier.py`) — keyword + retrieval heuristics for company inference, product area, and request type. No external model needed.
4. **TriageAgent** (`agent.py`) — wires all components. Calls Groq (llama-3.3-70b-versatile, temperature=0) only after escalation rules pass. Falls back to escalation if LLM returns empty.
5. **Prompts** (`prompts.py`) — all prompt strings in one file for easy tuning.

---

## Setup

```bash
cd code/
pip install -r requirements.txt

# Set up your API key
cp ../.env.example ../.env
# Edit ../.env and fill in GROQ_API_KEY
```

Get a free Groq API key at: https://console.groq.com

---

## Run

```bash
# Full run — writes support_tickets/output.csv
python main.py

# Validate against sample data first (recommended before full run)
python main.py --sample

# Custom paths
python main.py --input path/to/tickets.csv --output path/to/output.csv
```

---

## Design Decisions

**Why Groq + Llama instead of a paid API?**  
Groq's free tier is fast (tokens/sec), deterministic at temperature=0, and has no per-token cost ceiling during a hackathon. The model (llama-3.3-70b-versatile) follows instruction prompts reliably enough for structured support triage.

**Why local sentence-transformers for retrieval?**  
The problem statement forbids live web calls for ground-truth answers. Local embeddings with `all-MiniLM-L6-v2` run fully offline after initial model download, are deterministic, and produce an embedding cache so repeated runs are instant.

**Why HTML stripping?**  
The corpus contains `.html` files. Reading them raw injects thousands of `<div>`, `<script>`, `<a href=...>` tokens into every chunk, which destroys cosine similarity quality. BeautifulSoup strips all tags before chunking.

**Why rule-based escalation before the LLM?**  
Hard safety rules (prompt injection, malicious commands, fraud, score manipulation) should never reach an LLM. Running them first is cheaper, faster, and prevents the LLM from being tricked by adversarial tickets.

**Why escalate rather than guess?**  
The evaluation criteria explicitly penalise hallucinated policies. When retrieval returns no relevant chunks for a sensitive topic, escalating is the safe and correct choice.

---

## Known Limitations

- Product area classification is keyword + retrieval heuristics, not a trained classifier. Uncommon phrasings may mis-classify.
- If the corpus is missing documentation for a topic, the agent will escalate conservatively — this is the intended behaviour.
- Non-English issues are treated as suspicious if they request system internals; legitimate non-English support tickets may be over-escalated.
- Groq rate limits may slow down runs with many tickets — add `time.sleep(0.2)` to `main.py` if you hit 429 errors.