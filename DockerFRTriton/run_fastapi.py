"""
Helper script to set up a local venv, install requirements, and launch the FastAPI wrapper.

Usage:
    python run_fastapi.py
"""

import subprocess
import sys
import venv
from pathlib import Path


VENV_DIR = Path(__file__).parent / ".venv"
REQ_FILE = Path(__file__).parent / "requirements.txt"


def ensure_venv() -> Path:
    """Create .venv with pip if missing; return path to venv python."""
    if not VENV_DIR.exists():
        print(f"[setup] Creating virtual environment at {VENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    python_bin = VENV_DIR / "bin" / "python"
    if not python_bin.exists():  # Windows fallback
        python_bin = VENV_DIR / "Scripts" / "python.exe"
    if not python_bin.exists():
        raise FileNotFoundError("Could not find python inside the virtual environment.")
    return python_bin


def install_requirements(python_bin: Path) -> None:
    """Install Python dependencies into the venv."""
    print(f"[setup] Installing requirements from {REQ_FILE}")
    subprocess.check_call([str(python_bin), "-m", "pip", "install", "-r", str(REQ_FILE)])


def launch_uvicorn(python_bin: Path) -> None:
    """Run uvicorn using the venv python."""
    cmd = [
        str(python_bin),
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "5004",
    ]
    print(f"[run] Launching: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    python_bin = ensure_venv()
    install_requirements(python_bin)
    try:
        launch_uvicorn(python_bin)
    except KeyboardInterrupt:
        print("\n[run] Stopped by user.")


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        raise SystemExit("Python 3.8+ is required.")
    main()
