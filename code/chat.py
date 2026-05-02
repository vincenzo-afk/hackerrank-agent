#!/usr/bin/env python3
"""
Interactive CLI Chat Agent — HackerRank Orchestrate
====================================================
Run:  python code/chat.py
      python code/chat.py --company hackerrank   # pre-set company context
      python code/chat.py --reindex              # force rebuild embedding cache
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import pandas as pd
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


COMPANY_COLOURS = {
    "hackerrank": green,
    "claude": magenta,
    "visa": blue,
    None: dim,
}

VALID_COMPANIES = {"hackerrank", "claude", "visa"}


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
        sys.stdout.write("\r" + " " * (len(self._label) + 8) + "\r")
        sys.stdout.flush()

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            sys.stdout.write(f"\r  {cyan(frame)} {dim(self._label + '…')}  ")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
BANNER = f"""
{cyan('╔═══════════════════════════════════════════════════╗')}
{cyan('║')}  {bold('HackerRank Orchestrate — Support Chat')}            {cyan('║')}
{cyan('║')}  {dim('Domains:')} {green('HackerRank')} · {magenta('Claude')} · {blue('Visa')}               {cyan('║')}
{cyan('║')}  {dim('Type')} {yellow('help')} {dim('for commands or just ask a question.')}  {cyan('║')}
{cyan('╚═══════════════════════════════════════════════════╝')}
"""

HELP_TEXT = f"""
{bold(cyan('Commands'))}
  {yellow('help')}                     Show this help message
  {yellow('clear')}                    Clear the screen
  {yellow('history')}                  Show conversation history
  {yellow('company')}                  Show current company context
  {yellow('company <name>')}           Set company (hackerrank / claude / visa / none)
  {yellow('quit')} / {yellow('exit')}              Exit the agent

{bold(cyan('How to use'))}
  Just type your support question and press Enter.
  The agent will retrieve relevant docs and reply conversationally.

{bold(cyan('Company context'))}
  Setting a company helps the agent retrieve more targeted answers.
  If not set, the agent will auto-detect from your question.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _company_label(company: str | None) -> str:
    fn = COMPANY_COLOURS.get(company, dim)
    return fn(f"[{company}]") if company else dim("[auto]")


def _wrap(text: str, width: int = 72, indent: str = "    ") -> str:
    """Wrap long text at word boundaries, preserving blank lines."""
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


def _print_user_turn(text: str, company: str | None) -> None:
    label = _company_label(company)
    print(f"\n  {bold(cyan('You'))} {label}")
    print(f"  {dim('›')} {text}\n")


def _print_agent_turn(result: dict) -> None:
    status = result.get("status", "")
    area = result.get("product_area", "")
    req_type = result.get("request_type", "")
    response = result.get("response", "")
    justification = result.get("justification", "")

    # We want a 52 character width bar:
    bar = cyan("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {bar}")
    print(f"  {bold('STATUS')}       : {status}")
    print(f"  {bold('PRODUCT AREA')} : {area}")
    print(f"  {bold('REQUEST TYPE')} : {req_type}")
    
    resp_lines = textwrap.fill(response, width=52, subsequent_indent="                 ")
    print(f"  {bold('RESPONSE')}     : {resp_lines}")
    
    just_lines = textwrap.fill(justification, width=52, subsequent_indent="                 ")
    print(f"  {bold('JUSTIFICATION')}: {just_lines}")
    print(f"  {bar}")
    print()


# ---------------------------------------------------------------------------
# ChatSession
# ---------------------------------------------------------------------------

class ChatSession:
    def __init__(self, preset_company: str | None = None) -> None:
        self.company_override: str | None = preset_company
        self.history: list[dict] = []          # display history (role/text)
        self.llm_history: list[dict] = []      # OpenAI-format history passed to LLM
        self._turn = 0
        self._retriever = None
        self._agent = None

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    def boot(self, force_reindex: bool = False) -> None:
        load_dotenv()

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from retriever import CorpusRetriever
        from agent import TriageAgent

        print(f"\n  {dim('Loading corpus and building embedding index…')}")
        if force_reindex:
            print(f"  {yellow('!')} {dim('Force reindex — this will take several minutes.')}")
        else:
            print(f"  {dim('(Cached index loads in seconds. Use --reindex to rebuild.)')}")
        print()

        spinner = Spinner("Indexing corpus")
        spinner.start()
        boot_ok = False
        try:
            self._retriever = CorpusRetriever()
            self._retriever.load_and_index(force_reindex=force_reindex)
            self._agent = TriageAgent(self._retriever)
            boot_ok = True
        except Exception as e:
            spinner.stop()  # stop before printing error so line is clean
            print(f"\n  {red('✗')} Failed to boot: {e}\n")
            sys.exit(1)
        finally:
            if boot_ok:   # only stop spinner if we didn't stop it in the except block
                spinner.stop()

        print(f"  {green('✓')} Ready. {dim('Type')} {yellow('help')} {dim('for commands.')}\n")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_help(self) -> None:
        print(HELP_TEXT)

    def _cmd_clear(self) -> None:
        os.system("clear" if os.name != "nt" else "cls")
        print(BANNER)

    def _cmd_history(self) -> None:
        if not self.history:
            print(f"\n  {dim('No history yet.')}\n")
            return
        print(f"\n  {bold('Conversation history')} ({len(self.history)} turns)\n")
        for i, h in enumerate(self.history, 1):
            role = bold(cyan("You")) if h["role"] == "user" else bold(green("Agent"))
            snippet = h["text"][:90] + ("…" if len(h["text"]) > 90 else "")
            print(f"  {dim(f'[{i}]')} {role}  {snippet}")
        print()

    def _cmd_company(self, arg: str) -> None:
        arg = arg.strip().lower()
        if arg in {"none", "reset", "clear", ""}:
            self.company_override = None
            print(f"\n  {green('✓')} Company context cleared — agent will auto-detect.\n")
        elif arg in VALID_COMPANIES:
            self.company_override = arg
            fn = COMPANY_COLOURS.get(arg, dim)
            print(f"\n  {green('✓')} Company set to {fn(arg)}.\n")
        else:
            print(f"\n  {yellow('!')} Unknown company '{arg}'. Valid: {', '.join(sorted(VALID_COMPANIES))}, none.\n")

    def _dispatch(self, raw: str) -> bool:
        """Returns True if handled as a command, False if it's a user question."""
        lower = raw.strip().lower()

        if lower in {"quit", "exit", "q", ":q", "bye", "goodbye"}:
            print(f"\n  {dim('Goodbye! 👋')}\n")
            sys.exit(0)

        if lower in {"help", "?"}:
            self._cmd_help()
            return True

        if lower in {"clear", "cls"}:
            self._cmd_clear()
            return True

        if lower in {"history", "hist", "h"}:
            self._cmd_history()
            return True

        if lower == "company":
            current = self.company_override or "auto-detect"
            fn = COMPANY_COLOURS.get(self.company_override, dim)
            print(f"\n  Current company: {fn(current)}\n")
            return True

        if lower.startswith("company "):
            self._cmd_company(raw.strip()[8:])
            return True

        return False

    # ------------------------------------------------------------------
    # Ticket processing
    # ------------------------------------------------------------------

    def _triage(self, text: str) -> dict:
        row = pd.Series({
            "Issue": text,
            "Subject": "",
            "Company": self.company_override or "",
        })
        return self._agent.process_ticket(row, conversation_history=self.llm_history)

    # ------------------------------------------------------------------
    # Main REPL loop
    # ------------------------------------------------------------------

    def run(self, force_reindex: bool = False) -> None:
        print(BANNER)
        if self.company_override:
            fn = COMPANY_COLOURS.get(self.company_override, dim)
            print(f"  Company context: {fn(self.company_override)}\n")

        self.boot(force_reindex=force_reindex)

        while True:
            try:
                raw = input(f"{bold(cyan('  ›  '))}").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {dim('Interrupted. Goodbye! 👋')}\n")
                break

            if not raw:
                continue

            if self._dispatch(raw):
                continue

            # It's a support question
            self._turn += 1
            _print_user_turn(raw, self.company_override)
            self.history.append({"role": "user", "text": raw})

            spinner = Spinner("Thinking")
            spinner.start()
            try:
                result = self._triage(raw)
            except Exception as exc:
                spinner.stop()
                print(f"  {red('✗')} Error: {exc}\n")
                continue
            spinner.stop()

            response = result.get("response", "Sorry, I could not generate a response.")

            _print_agent_turn(result)

            # Store for display
            self.history.append({"role": "agent", "text": response})
            # Store in LLM format for multi-turn memory
            self.llm_history.append({"role": "user", "content": raw})
            self.llm_history.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HackerRank Orchestrate — Interactive Support Chat Agent"
    )
    parser.add_argument(
        "--company",
        choices=list(VALID_COMPANIES),
        default=None,
        help="Pre-set company context",
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
