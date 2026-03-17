#!/usr/bin/env python3
"""
Segment Clickety Split — GUI Launcher
======================================
Double-click entry point for the PyInstaller executable.

- If a project directory is passed as an argument, uses it directly.
- Otherwise, opens a folder-picker dialog (requires tkinter, which is
  bundled with Python on Windows/macOS).
- Auto-opens the browser after the server starts.
"""

import sys
import threading
import time
import webbrowser
from pathlib import Path


def pick_folder():
    """Show a native folder-picker dialog and return the chosen path."""
    import subprocess

    # Try Windows native folder picker via PowerShell
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
        "$d.Description = 'Select your recordings folder'; "
        "$d.UseDescriptionForTitle = $true; "
        "$d.ShowNewFolderButton = $false; "
        "[void]$d.ShowDialog(); "
        "$d.SelectedPath"
    )
    for ps_cmd in (['powershell.exe', '-NoProfile', '-Command', ps],  # WSL → Windows
                   ['powershell',     '-NoProfile', '-Command', ps]):  # native Windows
        try:
            result = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=300)
            folder = result.stdout.strip()
            if not folder:
                sys.exit(0)
            # On WSL, convert Windows path (C:\...) to POSIX path (/mnt/c/...)
            try:
                conv = subprocess.run(['wslpath', folder], capture_output=True, text=True)
                if conv.returncode == 0 and conv.stdout.strip():
                    folder = conv.stdout.strip()
            except Exception:
                pass
            return folder
        except FileNotFoundError:
            continue
        except Exception:
            break

    # Fallback: tkinter
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        folder = filedialog.askdirectory(
            title="Select your recordings folder"
        )
        root.destroy()
        if folder:
            return folder
        sys.exit(0)
    except Exception:
        pass

    # Last resort: prompt
    folder = input("Enter project directory path: ").strip()
    if not folder:
        sys.exit(0)
    return folder


def open_browser(url: str, delay: float = 1.5):
    """Open the browser after a short delay so Flask has time to bind."""
    time.sleep(delay)
    webbrowser.open(url)


def main():
    # ── 1. Resolve project directory ───────────────────────────────────────
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        project_dir = pick_folder()

    project_path = Path(project_dir).resolve()
    if not project_path.is_dir():
        print(f"ERROR: '{project_path}' is not a directory.")
        input("Press Enter to exit...")
        sys.exit(1)

    # ── 2. Configure app globals before importing ───────────────────────────
    # Patch sys.argv so app.py's argparse sees the right arguments
    sys.argv = ["app.py", str(project_path)]

    # ── 3. Start browser opener in background ──────────────────────────────
    port = 5000
    url = f"http://127.0.0.1:{port}"
    t = threading.Thread(target=open_browser, args=(url,), daemon=True)
    t.start()

    # ── 4. Run the Flask app ────────────────────────────────────────────────
    print(f"\n  Segment Clickety Split")
    print(f"  {url}")
    print(f"  Project: {project_path}")
    print(f"\n  Opening browser automatically...")
    print(f"  (If the browser doesn't open, go to {url})")
    print(f"\n  Press Ctrl+C to quit.\n")

    import app as flask_app  # noqa: E402  (import after path setup)

    # If the selected folder contains a "recordings" subfolder, treat it as
    # the project root (legacy layout).  Otherwise treat the folder itself as
    # the recordings directory and store experiment data alongside it.
    if (project_path / "recordings").is_dir():
        recordings_dir = project_path / "recordings"
        experiment_dir = project_path / "experiment"
        sessions_dir   = project_path / "sessions"
    else:
        recordings_dir = project_path
        experiment_dir = project_path.parent / (project_path.name + "_experiment")
        sessions_dir   = project_path.parent / (project_path.name + "_sessions")

    flask_app.PROJECT_ROOT   = project_path
    flask_app.RECORDINGS_DIR = recordings_dir
    flask_app.EXPERIMENT_DIR = experiment_dir
    flask_app.STIMLISTS_DIR  = experiment_dir / "stimulus_lists"
    flask_app.SESSIONS_DIR   = sessions_dir
    flask_app.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    flask_app.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
