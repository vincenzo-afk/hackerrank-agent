from __future__ import annotations

import argparse

from dotenv import load_dotenv
from tqdm import tqdm

from agent import TriageAgent
from retriever import CorpusRetriever
from utils import load_tickets, write_output


def main() -> None:
    load_dotenv()
    retriever = CorpusRetriever()
    retriever.load_and_index()
    agent = TriageAgent(retriever)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="support_tickets/support_tickets.csv",
        help="Path to input tickets CSV (default: support_tickets/support_tickets.csv)",
    )
    parser.add_argument(
        "--output",
        default="output.csv",
        help="Path to output CSV (default: output.csv at repo root)",
    )
    args = parser.parse_args()

    tickets = load_tickets(args.input)
    results: list[dict] = []

    for idx, (_, row) in enumerate(tqdm(tickets.iterrows(), total=len(tickets))):
        result = agent.process_ticket(row)
        results.append(result)
        company = row.get("Company", "")
        print(f"[{idx+1}/{len(tickets)}] {company} | {result['status']} | {result['product_area']}")

    write_output(results, args.output)
    print(f"Done. Output written to {args.output}")


if __name__ == "__main__":
    main()
