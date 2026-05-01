# Multi-Domain Support Triage Agent — code/

## Overview

Terminal-based triage agent for HackerRank, Claude, and Visa support tickets.
Supports two modes:

| Mode | Command | Description |
|------|---------|-------------|
| **Interactive Chat** | `python code/chat.py` | Live REPL — type questions, get instant answers |
| **Batch CSV run** | `python code/main.py` | Process `support_tickets/support_tickets.csv` → `output.csv` |

Output columns: `status, product_area, response, justification, request_type`

---

## Quick Start

### 1. Install dependencies

```bash
cd code/
pip install -r requirements.txt
```

> **Note:** Requires Python 3.9–3.12. Python 3.13+ may have torch compatibility issues.
> Use `/usr/bin/python3 -m venv venv && source venv/bin/activate` if the default python is too new.

### 2. Set your Groq API key

```bash
cp ../.env.example ../.env
# Edit ../.env and fill in:
# GROQ_API_KEY=gsk_...
```

Get a free Groq API key at: https://console.groq.com

---

## Interactive Chat Mode (NEW)

```bash
# Start the chat agent (from repo root)
source code/venv_old/bin/activate
python code/chat.py

# Pre-set company context to skip auto-detection
python code/chat.py --company hackerrank
python code/chat.py --company claude
python code/chat.py --company visa
```

### Chat commands

| Command | Action |
|---------|--------|
| `help` | Show available commands |
| `clear` | Clear the screen |
| `history` | Show conversation history |
| `company <name>` | Set company context (`hackerrank` / `claude` / `visa` / `none`) |
| `quit` / `exit` | Exit the agent |

### Example session

```
  ›  I can't log in to my HackerRank account

  ╔ REPLIED ╗  account  ·  product_issue
  ─────────────────────────────────────────────────────────────
    To reset your password, go to the HackerRank login page
    and click "Forgot your password?"…
```

---

## Batch CSV Mode

```bash
# Full run — writes support_tickets/output.csv
python code/main.py

# Quick validation against sample data first
python code/main.py --sample

# Custom input/output paths
python code/main.py --input path/to/tickets.csv --output path/to/output.csv
```

---

## Architecture

`main.py` / `chat.py` orchestrate the pipeline:

1. **CorpusRetriever** (`retriever.py`) — walks `data/`, strips HTML (BeautifulSoup), parses JSON, chunks into ~1200-char overlapping windows, embeds with `all-MiniLM-L6-v2`, caches to `data/embeddings_cache.npy`.
2. **EscalationEngine** (`escalation.py`) — rule-based pre-checks (prompt injection, malicious commands, fraud, identity theft, score manipulation, outages) run *before* any LLM call.
3. **Classifier** (`classifier.py`) — keyword + retrieval heuristics for company inference, product area, and request type.
4. **TriageAgent** (`agent.py`) — wires all components. Calls Groq (llama-3.3-70b-versatile, temperature=0) only after escalation rules pass.
5. **Prompts** (`prompts.py`) — all prompt strings in one place for easy tuning.
6. **ChatSession** (`chat.py`) — interactive REPL with coloured output, spinner, history, and company context switching.

---

## Design Decisions

**Why Groq + Llama?**
Groq's free tier is fast, deterministic at temperature=0, and has no per-token cost ceiling for a hackathon.

**Why local sentence-transformers?**
Fully offline after initial model download. Deterministic. Embedding cache makes repeated runs instant.

**Why HTML stripping?**
Raw HTML corpus files inject tag noise that destroys cosine similarity. BeautifulSoup strips all tags before chunking.

**Why rule-based escalation before the LLM?**
Hard safety rules (injection, fraud, malicious commands) should never reach an LLM. Faster, cheaper, and adversarially robust.

**Why escalate rather than guess?**
Evaluation criteria penalise hallucinated policies. No corpus coverage → escalate.

---

## Known Limitations

- Product area classification is keyword + retrieval heuristics. Uncommon phrasing may mis-classify.
- If the corpus lacks documentation for a topic, the agent escalates conservatively (intended behaviour).
- Non-English injection attempts are flagged; legitimate non-English tickets may be over-escalated.
- Groq rate limits may slow large batch runs — add `time.sleep(0.2)` in `main.py` if you hit 429 errors.
- First run takes ~20 minutes to build the embedding index; subsequent runs use the cache and start instantly.