
import subprocess
import os
import sys

def kill_ghosts():
    print("\n" + "="*45)
    print("      NUCLEAR CLEANUP - KILLING ALL PROCESSES")
    print("="*45)

    my_pid = os.getpid()

    # Define cleanup tasks
    # /F = Force, /T = Kill child processes as well, /IM = Image Name (Wildcard ok)
    # /FI "PID ne ..." avoids self-suicide for the python command.
    
    commands = [
        # 1. Kill Stockfish engines
        'taskkill /F /IM "stockfish*" /T 2>nul',
        'taskkill /F /IM "lc0*" /T 2>nul',
        
        # 2. Kill Zchezz engines
        'taskkill /F /IM "zchezz*" /T 2>nul',
        
        # 3. Kill PowerShell instances (useful for orphans)
        'taskkill /F /IM "powershell.exe" /T 2>nul',
        
        # 4. Kill Node.js (bridges/engines)
        'taskkill /F /IM "node.exe" /T 2>nul',
        
        # 5. Kill Python processes (Targeting orphans, avoiding self-kill)
        f'taskkill /F /FI "PID ne {my_pid}" /IM "python.exe" /T 2>nul'
    ]

    for cmd in commands:
        try:
            # We use shell=True so taskkill executes directly in CMD shell
            subprocess.run(cmd, shell=True, capture_output=False)
        except Exception:
            pass

    print("\n[OK] Cleanup Complete. (Environment is now clean)")
    print("="*45 + "\n")

if __name__ == "__main__":
    kill_ghosts()
