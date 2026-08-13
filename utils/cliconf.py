"""
utils/cliconf.py — "config block at the top, CLI as an override" in one call
============================================================================

Every tool in this repo follows CLAUDE.md rule 8: the real defaults live in a
documented CONFIGURATION block at the TOP of the file, and the command line
only OVERRIDES them. Editing the constant and running `python tool.py` with
no arguments at all is the primary way to use these tools; the flags exist so
a scripted or one-off run does not have to edit the file.

Writing that twice per option (a constant, then an `add_argument(...,
default=THE_CONSTANT)`) is what this module removes. You declare each option
ONCE, next to its constant:

    # ═══════════════ CONFIGURATION ═══════════════
    GAMES       = 500      # number of games to play
    MOVETIME_MS = 100      # per-move budget, milliseconds
    SAVE_PGN    = False    # write a PGN of every game
    OPENINGS    = ["openings/lines"]   # folders searched for opening files
    # ═════════════════════════════════════════════

    CLI = [
        ("GAMES",       "--games",     int,  "number of games to play"),
        ("MOVETIME_MS", "--movetime",  int,  "per-move budget, ms"),
        ("SAVE_PGN",    "--pgn",       bool, "write a PGN of every game"),
        ("OPENINGS",    "--openings",  list, "opening folder (repeatable)"),
    ]

    if __name__ == "__main__":
        override_from_cli(globals(), CLI, description=__doc__)
        main()

and get, for free:

  * `--games N`, `--movetime N`
  * `--pgn` / `--no-pgn` for every bool (both directions always exist, so a
    constant that defaults to True can still be turned off)
  * `--openings X --openings Y` for every list (repeating REPLACES the
    constant's list rather than appending to it — appending to a non-empty
    default is a classic argparse trap)
  * `--help` listing every option WITH the value the config block gives it,
    so the help text can never drift from the actual default
  * `--show-config`, which prints the resolved settings and exits — the fast
    way to check what a bare run would do before starting a long job

The defaults are read from the mapping you pass in (normally `globals()`),
so the constant really is the single source of truth — this module never
holds a second copy of a literal. Parsed values are written BACK into that
same mapping, which is why the call belongs immediately after the
configuration block and BEFORE anything derived from it (timestamps, output
paths, os.makedirs, validation) is computed.

SHARED CONFIGURATION VOCABULARY
───────────────────────────────
One name per concept, the same in every tool, so a value means the same
thing whichever file you are reading. Use these names; invent a new one only
for something genuinely new to that tool.

  Games and search
    GAMES            number of games to play
    CONCURRENCY      games played in parallel — NOT threads inside one game.
                     0 = autodetect cores, in the tools that wrap a C exe
    MOVETIME_MS      per-move budget, milliseconds
    NODES            per-move node budget; used only when MOVETIME_MS == 0
    DEPTH            per-move search depth; 0 = use MOVETIME_MS/NODES instead
    MAX_PLIES        half-move cap before a game is scored a draw (400 here)
    MOVE_TIMEOUT_S   seconds to wait for one bestmove before calling it a loss
    TT_MB            per-TTable memory budget, MB

  Engines and paths
    ENGINE_EXE       the engine binary under test
    VERSION          engine folder suffix, as in engine/c/zchezz_<VERSION>
    SF_PATH          Stockfish binary, when one is used as anchor or labeller
    OPENING_FOLDER   walked recursively for .pgn/.epd opening files
    RESULTS_DIR      tests/<tool>_results — every harness writes under tests/
    DATA_DIR         root holding one folder per dataset
    OUT_DIR          single output folder for a generator

  Output switches (independent, never a choice of one)
    SAVE_PGN         write standard PGN game records
    SAVE_EPD         write positions + eval as EPD
    SAVE_BIN         write packed training samples (train/dataset.py SAMPLE_DTYPE)

  Runs and repeatability
    SEED             RNG seed; a fixed seed means a reproducible run
    DRY_RUN          do everything except write; report what would happen
    ONLY             restrict the run to these items ([] = all of them)
    WORKERS          worker processes/threads doing the CPU work
    BATCH_SIZE       items handed to one worker task
    CHUNK_SIZE_SAVE  rows per output chunk
    TIMEOUT_S        per-item wall-clock limit

A `DEFAULT_` prefix marks a knob specific to ONE tool and not shared.

TYPES
─────
  int / float / str  scalar, `--flag VALUE`
  bool               `--flag` / `--no-flag` pair
  list               repeatable `--flag V`; first use clears the default
  a callable         used as argparse's `type=` (e.g. a parse_limit()), so a
                     tool can accept its own richer syntax
  a tuple/list of strings   an enumeration -> argparse `choices=`
"""

from __future__ import annotations

import argparse
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


def force_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 tolerant.

    Windows consoles default to cp1252 and raise UnicodeEncodeError on the
    em-dashes and box characters used throughout this repo's help text and
    reports — the tool then dies before printing anything. Same guard as
    tests/bench_nps.py's, kept in one place so every tool can call it.
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass    # already redirected, or not reconfigurable


class _ReplacingAppend(argparse.Action):
    """`--flag a --flag b` -> ['a','b'], REPLACING the config default.

    Plain `action='append'` appends to whatever `default=` holds, so a
    non-empty config default can never be removed from the command line —
    you would always get the constant's entries plus yours. This clears the
    default the first time the flag is seen.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if not getattr(namespace, f"_{self.dest}_seen", False):
            current = []
            setattr(namespace, f"_{self.dest}_seen", True)
        current = list(current or [])
        current.append(values)
        setattr(namespace, self.dest, current)


def _flag_to_dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def build_parser(config: Mapping[str, Any], spec: Sequence[tuple],
                 prog: str | None = None, description: str | None = None,
                 epilog: str | None = None) -> argparse.ArgumentParser:
    """Parser whose every default IS the corresponding config constant."""
    ap = argparse.ArgumentParser(
        prog=prog, description=description, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-config", action="store_true",
                    help="print the resolved configuration and exit "
                         "(nothing is run, nothing is written)")

    for entry in spec:
        name, flag, kind, help_text = entry[0], entry[1], entry[2], entry[3]
        extra = entry[4] if len(entry) > 4 else None
        if name not in config:
            raise KeyError(f"cliconf: {name!r} is in the CLI spec but not in the "
                           f"configuration block — every flag needs a documented "
                           f"constant (CLAUDE.md rule 8)")
        default = config[name]
        dest    = _flag_to_dest(flag)
        shown   = f"(default: {default!r})"

        if kind is bool:
            g = ap.add_mutually_exclusive_group()
            g.add_argument(flag, dest=dest, action="store_true", default=default,
                           help=f"{help_text} {shown}")
            neg = "--no-" + flag.lstrip("-")
            g.add_argument(neg, dest=dest, action="store_false",
                           help=f"disable: {help_text}")
        elif kind is list:
            ap.add_argument(flag, dest=dest, action=_ReplacingAppend, default=list(default),
                            metavar="VALUE",
                            help=f"{help_text}; repeatable, replaces the default {shown}")
        else:
            kwargs: dict = {"dest": dest, "default": default,
                            "help": f"{help_text} {shown}"}
            if isinstance(extra, (tuple, list)):
                kwargs["choices"] = tuple(extra)
            if kind is not None:
                kwargs["type"] = kind
            ap.add_argument(flag, **kwargs)
    return ap


def override_from_cli(config: MutableMapping[str, Any], spec: Sequence[tuple],
                      argv: Sequence[str] | None = None,
                      prog: str | None = None, description: str | None = None,
                      epilog: str | None = None) -> argparse.Namespace:
    """Parse `argv` against `spec` and write the results back into `config`.

    `config` is normally the caller's `globals()`. Call this right after the
    CONFIGURATION block and before anything derived from it, so a flag and an
    edited constant reach the rest of the file by the same route.
    """
    ap   = build_parser(config, spec, prog=prog, description=description, epilog=epilog)
    args = ap.parse_args(argv)

    for entry in spec:
        name, flag = entry[0], entry[1]
        config[name] = getattr(args, _flag_to_dest(flag))

    if getattr(args, "show_config", False):
        width = max(len(e[0]) for e in spec)
        print("Resolved configuration:")
        for entry in spec:
            print(f"  {entry[0].ljust(width)} = {config[entry[0]]!r}")
        raise SystemExit(0)
    return args


def print_config(config: Mapping[str, Any], spec: Sequence[tuple], prefix: str = "  ") -> None:
    """Echo the resolved settings — handy at the start of a long run's log."""
    width = max(len(e[0]) for e in spec)
    for entry in spec:
        print(f"{prefix}{entry[0].ljust(width)} = {config[entry[0]]!r}")
