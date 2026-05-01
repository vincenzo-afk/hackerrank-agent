#!/usr/bin/env python3
"""
Interactive CLI Chat Agent — HackerRank Orchestrate
=====================================================
Run:  python code/chat.py
      python code/chat.py --company hackerrank   # pre-set company context
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# ANSI colour helpers (no extra deps — just sys.stdout checks)
# ---------------------------------------------------------------------------
def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


COLOUR = _supports_colour()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOUR else text


def bold(t: str) -> str:      return _c("1", t)
def dim(t: str) -> str:       return _c("2", t)
def green(t: str) -> str:     return _c("32", t)
def cyan(t: str) -> str:      return _c("36", t)
def yellow(t: str) -> str:    return _c("33", t)
def red(t: str) -> str:       return _c("31", t)
def magenta(t: str) -> str:   return _c("35", t)
def blue(t: str) -> str:      return _c("34", t)
def white_bg(t: str) -> str:  return _c("47;30", t)
def cyan_bg(t: str) -> str:   return _c("46;30", t)
def green_bg(t: str) -> str:  return _c("42;30", t)
def red_bg(t: str) -> str:    return _c("41;37", t)


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------
class Spinner:
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Thinking") -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        # Erase the spinner line
        sys.stdout.write("\r" + " " * (len(self._label) + 6) + "\r")
        sys.stdout.flush()


    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            sys.stdout.write(f"\r  {cyan(frame)} {dim(self._label)}…  ")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1


# ---------------------------------------------------------------------------
# Pretty banner
# ---------------------------------------------------------------------------
BANNER = f"""
{cyan('╔══════════════════════════════════════════════════════════╗')}
{cyan('║')}  {bold(white_bg('  HackerRank Orchestrate — Support Triage Chat Agent  '))}  {cyan('║')}
{cyan('║')}  Domains: {green('HackerRank')} · {magenta('Claude')} · {blue('Visa')}                          {cyan('║')}
{cyan('╠══════════════════════════════════════════════════════════╣')}
{cyan('║')}  {dim('Type your support question and press Enter.')}              {cyan('║')}
{cyan('║')}  {dim('Commands:')} {yellow('help')} · {yellow('clear')} · {yellow('history')} · {yellow('company <name>')} · {yellow('quit')}  {cyan('║')}
{cyan('╚══════════════════════════════════════════════════════════╝')}
"""


HELP_TEXT = f"""
{bold(cyan('Available commands:'))}
  {yellow('help')}              Show this help message
  {yellow('clear')}             Clear the screen
  {yellow('history')}           Show conversation history
  {yellow('company <name>')}    Set company context (hackerrank / claude / visa / none)
  {yellow('quit')} / {yellow('exit')}      Exit the agent

{bold(cyan('How it works:'))}
  Type any support question. The agent will:
  • Retrieve relevant documentation from the local corpus
  • Classify the request (product_area, request_type)
  • Either reply with a grounded answer or escalate to a human

{bold(cyan('Status indicators:'))}
  {green_bg(' REPLIED ')}    Answer generated from corpus
  {red_bg(' ESCALATED ')}  Requires human support specialist
"""

COMPANY_COLOURS = {
    "hackerrank": green,
    "claude": magenta,
    "visa": blue,
    None: dim,
}

# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _company_label(company: str | None) -> str:
    fn = COMPANY_COLOURS.get(str(company).lower() if company else None, dim)
    return fn(f"[{company or 'unknown'}]") if company else dim("[no company]")


def _wrap(text: str, width: int = 72, indent: str = "    ") -> str:
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        if line.strip():
            wrapped = textwrap.fill(line.strip(), width=width - len(indent))
            for wl in wrapped.splitlines():
                out.append(indent + wl)
        else:
            out.append("")
    return "\n".join(out)


def _print_reply(result: dict, idx: int) -> None:
    status = result.get("status", "?")
    area = result.get("product_area", "?")
    rtype = result.get("request_type", "?")
    response = result.get("response", "")
    justification = result.get("justification", "")

    if status == "replied":
        badge = green_bg(" REPLIED ")
        area_col = green(area)
    else:
        badge = red_bg(" ESCALATED ")
        area_col = red(area)

    print()
    print(f"  {badge}  {dim(f'#{idx}')}  {area_col}  {dim('·')}  {dim(rtype)}")
    print(f"  {dim('─' * 60)}")
    # Response
    print(_wrap(response))
    # Justification (dim)
    if justification:
        print()
        print(_wrap(dim(f"ℹ  {justification}")))
    print()


def _print_user(text: str, company: str | None) -> None:
    label = _company_label(company)
    print(f"\n  {bold(cyan('You'))} {label}  {dim('›')}  {text}")


# ---------------------------------------------------------------------------
# Main ChatSession
# ---------------------------------------------------------------------------

class ChatSession:
    def __init__(self, preset_company: str | None = None) -> None:
        self.company_override: str | None = preset_company
        self.history: list[dict] = []   # {role, text, result}
        self._turn = 0
        self._retriever = None
        self._agent = None

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    def boot(self, force_reindex: bool = False) -> None:
        """Load corpus and initialise agent (shown once at startup)."""
        load_dotenv()

        # Dynamic import so we only pay for it once
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from retriever import CorpusRetriever
        from agent import TriageAgent

        print(f"\n  {dim('Loading corpus and building embedding index…')}")
        if not force_reindex:
            print(f"  {dim('(This takes ~20 min on first run; subsequent runs use the cache.)')}")
        else:
            print(f"  {yellow('!')} {dim('Force reindexing enabled. This will take ~20 minutes.')}")
        print()

        spinner = Spinner("Indexing corpus")
        spinner.start()

        try:
            self._retriever = CorpusRetriever()
            self._retriever.load_and_index(force_reindex=force_reindex)
            self._agent = TriageAgent(self._retriever)
        finally:
            spinner.stop()

        print(f"  {green('✓')} Corpus indexed.  {dim('Type')} {yellow('help')} {dim('for commands.')}\n")

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _cmd_help(self, _: str) -> bool:
        print(HELP_TEXT)
        return True

    def _cmd_clear(self, _: str) -> bool:
        os.system("clear" if os.name != "nt" else "cls")
        print(BANNER)
        return True

    def _cmd_history(self, _: str) -> bool:
        if not self.history:
            print(f"\n  {dim('No conversation history yet.')}\n")
            return True
        print(f"\n  {bold('Conversation history')} ({len(self.history)} turns)\n")
        for i, h in enumerate(self.history, 1):
            role = "You" if h["role"] == "user" else "Agent"
            label = bold(cyan(role)) if h["role"] == "user" else bold(green(role))
            print(f"  [{i}] {label}: {h['text'][:100]}{'…' if len(h['text']) > 100 else ''}")
        print()
        return True

    def _cmd_company(self, arg: str) -> bool:
        arg = arg.strip().lower()
        if arg in {"none", "reset", "clear", ""}:
            self.company_override = None
            print(f"\n  {green('✓')} Company context cleared. Agent will auto-detect.\n")
        elif arg in {"hackerrank", "claude", "visa"}:
            self.company_override = arg
            fn = COMPANY_COLOURS.get(arg, dim)
            print(f"\n  {green('✓')} Company set to {fn(arg)}.\n")
        else:
            print(f"\n  {yellow('!')} Unknown company '{arg}'. Valid: hackerrank, claude, visa, none.\n")
        return True

    def _dispatch_command(self, raw: str) -> bool | None:
        """Returns True if handled as command, None if not a command."""
        stripped = raw.strip()
        lower = stripped.lower()

        if lower in {"quit", "exit", "q", ":q", "bye"}:
            print(f"\n  {dim('Goodbye! 👋')}\n")
            sys.exit(0)

        if lower in {"help", "?"}:
            return self._cmd_help(stripped)

        if lower in {"clear", "cls"}:
            return self._cmd_clear(stripped)

        if lower in {"history", "hist", "h"}:
            return self._cmd_history(stripped)

        if lower.startswith("company "):
            return self._cmd_company(stripped[8:])

        if lower == "company":
            current = self.company_override or "auto-detect"
            print(f"\n  Current company context: {bold(current)}\n")
            return True

        return None

    # ------------------------------------------------------------------
    # Ticket processing
    # ------------------------------------------------------------------

    def _process_message(self, text: str, subject: str = "") -> dict:
        """Run message through the triage agent and return the result dict."""
        import pandas as pd

        row = pd.Series(
            {
                "Issue": text,
                "Subject": subject,
                "Company": self.company_override or "",
            }
        )
        return self._agent.process_ticket(row)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, force_reindex: bool = False) -> None:
        print(BANNER)
        if self.company_override:
            fn = COMPANY_COLOURS.get(self.company_override, dim)
            print(f"  Company context pre-set: {fn(self.company_override)}\n")

        self.boot(force_reindex=force_reindex)

        prompt_prefix = cyan("  ›  ")

        while True:
            try:
                raw = input(f"{bold(prompt_prefix)}").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {dim('Interrupted. Goodbye! 👋')}\n")
                break

            if not raw:
                continue

            # Check for commands
            handled = self._dispatch_command(raw)
            if handled is True:
                continue

            # It's a support question — triage it
            self._turn += 1
            _print_user(raw, self.company_override)

            # Record in history
            self.history.append({"role": "user", "text": raw})

            spinner = Spinner("Thinking")
            spinner.start()

            try:
                result = self._process_message(raw)
            except Exception as exc:
                spinner.stop()
                print(f"\n  {red('✗')} Error processing ticket: {exc}\n")
                continue

            spinner.stop()
            _print_reply(result, self._turn)

            self.history.append({"role": "agent", "text": result.get("response", ""), "result": result})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HackerRank Orchestrate — Interactive CLI Support Chat Agent"
    )
    parser.add_argument(
        "--company",
        choices=["hackerrank", "claude", "visa"],
        default=None,
        help="Pre-set the company context (skips auto-detection)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild of the embedding index (ignores cache)",
    )
    args = parser.parse_args()

    session = ChatSession(preset_company=args.company)
    session.run(force_reindex=args.reindex)


if __name__ == "__main__":
    main()
