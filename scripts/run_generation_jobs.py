"""Run several scripts/generate.py jobs, one per GPU at a time.

    python scripts/run_generation_jobs.py --jobs outputs/eval/generation_jobs.json

The jobs file is a JSON list of objects whose keys are generate.py flags without the leading
dashes, e.g. {"model_path": ..., "generator": ..., "eval_set": ..., "output_path": ...}.
Each job's log goes to outputs/logs/gen_<output stem>.log. Exits non-zero if any job fails.
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alignment.utils import read_json  # noqa: E402


def job_command(job: dict, device: str) -> list[str]:
    command = [sys.executable, str(REPO_ROOT / "scripts" / "generate.py")]
    for key, value in job.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif value is not None:
            command += [f"--{key}", str(value)]
    return command + ["--device", device]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--num_gpus", type=int, default=None, help="Defaults to all visible GPUs (CPU if none).")
    parser.add_argument("--log_dir", default="outputs/logs")
    args = parser.parse_args()

    import torch

    jobs = read_json(args.jobs)
    num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(num_gpus)] or ["cpu"]
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    pending: queue.Queue = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures, lock, start = [], threading.Lock(), time.time()

    def worker(device: str) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            name = Path(job["output_path"]).stem
            log_path = Path(args.log_dir) / f"gen_{name}.log"
            with open(log_path, "w", encoding="utf-8") as log:
                code = subprocess.call(job_command(job, device), stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
            last_line = ([line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()] or [""])[-1]
            with lock:
                print(f"[{(time.time() - start) / 60:5.1f}m] {device} {'OK  ' if code == 0 else 'FAIL'} {name}: {last_line[:160]}", flush=True)
                if code != 0:
                    failures.append(name)

    threads = [threading.Thread(target=worker, args=(device,)) for device in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"{len(jobs) - len(failures)}/{len(jobs)} jobs succeeded" + (f"; failed: {failures}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
