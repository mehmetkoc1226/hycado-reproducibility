from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import traceback
from pathlib import Path

from .common import ROOT, system_info, write_json


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def prepare_logged(raw, output, reports):
    from .data import prepare
    reports = Path(reports).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    log_path = reports / "02_veri_hazirlama.log"
    quality_path = reports / "02_veri_kalite_raporu.json"
    # A failed attempt must not expose an older success report as current.
    quality_path.unlink(missing_ok=True)
    print(f"Gunluk dosyasi: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            try:
                prepared = prepare(raw, output)
                shutil.copy2(prepared / "data_quality.json", quality_path)
                print(f"Paylasilacak kalite raporu: {quality_path}")
            except Exception:
                print("DURDURULDU: Ayrintilar asagida ve gunluk dosyasinda.", file=sys.stderr)
                traceback.print_exc()
                return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="HyCADO temiz baslangic / yerel kapasite testi")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--torch", action="store_true")
    prep = sub.add_parser("prepare")
    prep.add_argument("--raw", default=str(ROOT / "data/raw"))
    prep.add_argument("--output", default=str(ROOT / "data/prepared"))
    prep.add_argument("--reports", default=str(ROOT / "reports"))
    test = sub.add_parser("probe")
    test.add_argument("--prepared", default=str(ROOT / "data/prepared"))
    test.add_argument("--reports", default=str(ROOT / "reports"))
    test.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    test.add_argument("--roles", nargs="+", choices=["yolo800", "rtdetr800", "rtdetr960"])
    test.add_argument("--precision", choices=["amp", "fp32"], default="amp")
    test.add_argument("--timeout", type=int, default=1800)
    longer = sub.add_parser("stability")
    longer.add_argument("--prepared", default=str(ROOT / "data/prepared"))
    longer.add_argument("--reports", default=str(ROOT / "reports"))
    longer.add_argument("--epochs", type=int, default=3)
    longer.add_argument("--train-images", type=int, default=256)
    longer.add_argument("--validation-images", type=int, default=32)
    longer.add_argument("--timeout", type=int, default=3600)
    real = sub.add_parser("stage1")
    real.add_argument("--prepared", default=str(ROOT / "data/prepared"))
    real.add_argument("--run-dir", default=str(ROOT / "experiments/stage1_v1_seed42"))
    internal = sub.add_parser("_probe_child", help=argparse.SUPPRESS)
    internal.add_argument("--job", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            info = system_info(args.torch)
            write_json(ROOT / "reports" / "01_python_cuda_kontrol.json", info)
            print(__import__("json").dumps(info, indent=2, ensure_ascii=False))
            if args.torch and not info.get("cuda_available", False):
                return 1
        elif args.command == "prepare":
            return prepare_logged(args.raw, args.output, args.reports)
        elif args.command == "probe":
            from .pilot import run
            if args.timeout <= 0:
                raise ValueError("Timeout pozitif olmali")
            run(args.prepared, args.reports, args.batches, args.roles, args.precision, args.timeout)
        elif args.command == "stability":
            from .stability import run
            _, completed = run(args.prepared, args.reports, args.epochs, args.train_images,
                               args.validation_images, args.timeout)
            return 0 if completed else 1
        elif args.command == "stage1":
            from .stage1 import run
            _, completed = run(args.prepared, args.run_dir)
            return 0 if completed else 1
        else:
            from .pilot import child
            return child(args.job)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print("DURDURULDU:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
