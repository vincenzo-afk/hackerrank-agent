from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from agent import TriageAgent
from retriever import CorpusRetriever
from utils import load_tickets, write_output

# Repo root is one level above code/
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT  = str(ROOT_DIR / "support_tickets" / "support_tickets.csv")
DEFAULT_OUTPUT = str(ROOT_DIR / "support_tickets" / "output.csv")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="HackerRank Orchestrate — Multi-Domain Support Triage Agent"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to input tickets CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample_support_tickets.csv instead (for validation)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild of the embedding index (ignores cache)",
    )
    args = parser.parse_args()

    # Allow --sample flag to override input to the sample file
    if args.sample:
        args.input = str(ROOT_DIR / "support_tickets" / "sample_support_tickets.csv")
        args.output = str(ROOT_DIR / "support_tickets" / "sample_output.csv")
        print("[main] Running on SAMPLE tickets for validation.")

    print("=" * 60)
    print("  HackerRank Orchestrate — Support Triage Agent")
    print("=" * 60)

    # Build retriever + index
    retriever = CorpusRetriever()
    retriever.load_and_index(force_reindex=args.reindex)

    # Build agent
    agent = TriageAgent(retriever)

    # Load tickets
    tickets = load_tickets(args.input)
    total = len(tickets)
    print(f"\n[main] Loaded {total} tickets from {args.input}")
    print(f"[main] Output → {args.output}\n")

    results: list[dict] = []
    status_counts: dict[str, int] = {"replied": 0, "escalated": 0}
    type_counts: dict[str, int] = {}

    for idx, (_, row) in enumerate(tqdm(tickets.iterrows(), total=total, desc="Triaging")):
        result = agent.process_ticket(row)
        results.append(result)

        # Track stats
        s = result.get("status", "?")
        t = result.get("request_type", "?")
        status_counts[s] = status_counts.get(s, 0) + 1
        type_counts[t] = type_counts.get(t, 0) + 1

        company = str(row.get("Company", "")).strip() or "None"
        area = result.get("product_area", "?")
        print(
            f"  [{idx+1:02d}/{total}] {company:12s} | {s:9s} | {area:25s} | {t}"
        )

    # Write output CSV
    write_output(results, args.output)

    # Summary
    print("\n" + "=" * 60)
    print(f"  Done. Output written to: {args.output}")
    print(f"  replied={status_counts.get('replied',0)}  "
          f"escalated={status_counts.get('escalated',0)}")
    print(f"  request_types: {type_counts}")
    print("=" * 60)


if __name__ == "__main__":
    main()