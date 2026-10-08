"""Integration tests ensuring all legacy examples execute correctly."""

from pathlib import Path
import os
import subprocess
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_DIR = REPO_ROOT / "examples" / "legacy"

LEGACY_SCRIPTS = [
    "ex0.py",
    "ex1.py",
    "ex2.py",
    "ex3.py",
    "ex4.py",
    "ex5.py",
    "ex6Henry.py",
    "ex7Elder.py",
    "ex8.py",
    "ex9multilayer.py",
    "ex10.py",
    "ex11.py",
]


def test_all_legacy_scripts_exist():
    """Verify that all expected legacy example scripts are present."""
    for script_name in LEGACY_SCRIPTS:
        script_path = LEGACY_DIR / script_name
        assert script_path.is_file(), f"Expected legacy script not found: {script_path}"


@pytest.mark.parametrize("script_name", LEGACY_SCRIPTS)
def test_legacy_example(script_name):
    """Execute a legacy example script end-to-end and check for success."""
    script_path = LEGACY_DIR / script_name

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=360,
    )

    assert proc.returncode == 0, (
        f"Execution of {script_name} failed with return code {proc.returncode}.\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )
