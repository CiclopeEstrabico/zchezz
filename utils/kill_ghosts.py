
import subprocess
import os
import sys

# ═══════════════ CONFIGURATION ═══════════════
# Edit these and run with no arguments; every one is also a CLI flag that
# overrides it (CLAUDE.md rule 8, see the COMMAND LINE block below).
#
# IMAGES is the list of process image names to terminate. It is a LIST, not a
# hardcoded command sequence, because "which processes count as ghosts"
# changes with what you are running — a Stockfish-anchored tournament leaves
# different orphans than a WASM browser test.
IMAGES = [
    "stockfish*",     # anchor engines
    "lc0*",           # anchor engines
    "zchezz*",        # our engine, every version
    "powershell.exe", # orphaned helper shells
    "node.exe",       # bridges / browser test servers
]
KILL_PYTHON = True    # also kill python.exe (EXCEPT this process). Turn this
                      # off when a long training run must survive the cleanup —
                      # train_nnue.py is a python.exe like any other.
DRY_RUN = False       # print the taskkill commands instead of running them
# ═══════════════════════════════════════════

# ═══════════════ COMMAND LINE ═══════════════
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cliconf import override_from_cli  # noqa: E402

CLI = [
    ("IMAGES",       "--image",     list, "process image name to kill (repeatable, replaces the default list)"),
    ("KILL_PYTHON",  "--kill-python", bool, "also kill other python.exe processes"),
    ("DRY_RUN",      "--dry-run",   bool, "print what would be killed, kill nothing"),
]


def kill_ghosts():
    print("\n" + "="*45)
    print("      NUCLEAR CLEANUP - KILLING ALL PROCESSES")
    print("="*45)

    my_pid = os.getpid()

    # Define cleanup tasks
    # /F = Force, /T = Kill child processes as well, /IM = Image Name (Wildcard ok)
    # /FI "PID ne ..." avoids self-suicide for the python command.
    
    commands = [f'taskkill /F /IM "{img}" /T 2>nul' for img in IMAGES]
    if KILL_PYTHON:
        # /FI "PID ne ..." avoids self-suicide for this python process.
        commands.append(f'taskkill /F /FI "PID ne {my_pid}" /IM "python.exe" /T 2>nul')

    for cmd in commands:
        if DRY_RUN:
            print("  [dry-run]", cmd)
            continue
        try:
            # We use shell=True so taskkill executes directly in CMD shell
            subprocess.run(cmd, shell=True, capture_output=False)
        except Exception:
            pass

    print("\n[OK] Cleanup Complete. (Environment is now clean)")
    print("="*45 + "\n")

if __name__ == "__main__":
    override_from_cli(globals(), CLI, description=
                      "Kill leftover engine/helper processes from a crashed run.",
                      prog="kill_ghosts.py")
    kill_ghosts()
