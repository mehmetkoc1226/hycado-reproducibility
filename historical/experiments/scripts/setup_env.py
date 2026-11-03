"""Create a project-local venv; leave the user's global Python packages alone."""
from pathlib import Path
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent.parent


def main():
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise SystemExit("Python 3.11 veya 3.12 gerekli. Ilk donanim raporunu gonderin.")
    folder = ROOT / ".venv"
    py = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        venv.EnvBuilder(with_pip=True).create(folder)
    commands = [
        [str(py), "-m", "pip", "install", "--upgrade", "pip"],
        [str(py), "-m", "pip", "install", "torch==2.6.0", "torchvision==0.21.0",
         "--index-url", "https://download.pytorch.org/whl/cu124"],
        [str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements-pilot.txt")],
        [str(py), "-m", "hycado_start", "check", "--torch"],
        [str(py), "-m", "unittest", "discover", "-s", "tests", "-v"],
    ]
    for command in commands:
        print("\nCALISIYOR:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    with (report_dir / "installed_packages.txt").open("w", encoding="utf-8") as stream:
        subprocess.run([str(py), "-m", "pip", "freeze"], stdout=stream, check=True)
    print("\nKURULUM VE CUDA KONTROLU TAMAMLANDI.")


if __name__ == "__main__":
    main()
