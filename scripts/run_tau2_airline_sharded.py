from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_task_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(" ", ",").split(",") if item.strip()]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv

    if os.name == "nt":
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            candidate = (
                Path(user_profile)
                / "AppData"
                / "Roaming"
                / "Python"
                / "Python312"
                / "Scripts"
                / "uv.exe"
            )
            if candidate.exists():
                return str(candidate)

    raise SystemExit("uv not found. Install it first, e.g. `py -m pip install --user uv` or `brew install uv`.")


def completed_shard(simulations_root: Path, save_to: str) -> bool:
    results_json = simulations_root / save_to / "results.json"
    if not results_json.exists():
        return False
    try:
        import json

        data = json.loads(results_json.read_text(encoding="utf-8"))
        return bool(data.get("simulations"))
    except Exception:
        return False


def popen_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def stop_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def build_tau2_command(args: argparse.Namespace, uv: str, task_id: str, trial: int) -> tuple[list[str], int, str]:
    shard_seed = args.seed + trial
    save_to = f"{args.shard_prefix}_task_{task_id}_trial_{trial}"
    cmd = [
        uv,
        "run",
        "tau2",
        "run",
        "--domain",
        "airline",
        "--agent",
        "llm_agent",
        "--user",
        "user_simulator",
        "--agent-llm",
        args.agent_llm,
        "--user-llm",
        args.user_llm,
        "--num-trials",
        "1",
        "--max-concurrency",
        "1",
        "--max-steps",
        str(args.max_steps),
        "--max-retries",
        str(args.max_retries),
        "--retry-delay",
        str(args.retry_delay),
        "--seed",
        str(shard_seed),
        "--save-to",
        save_to,
        "--auto-resume",
        "--task-ids",
        str(task_id),
    ]
    if args.agent_llm_args:
        cmd += ["--agent-llm-args", args.agent_llm_args]
    if args.user_llm_args:
        cmd += ["--user-llm-args", args.user_llm_args]
    if args.verbose_logs:
        cmd += ["--verbose-logs", "--llm-log-mode", "latest"]
    return cmd, shard_seed, save_to


def append_manifest(manifest_path: Path, row: dict[str, object]) -> None:
    exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "task_id",
                "trial",
                "seed",
                "status",
                "exit_code",
                "duration_seconds",
                "save_to",
                "stdout",
                "stderr",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_shards(args: argparse.Namespace) -> None:
    root = repo_root()
    tau2_root = root / "third_party" / "tau2-bench"
    simulations_root = tau2_root / "data" / "simulations"
    reports_dir = root / "reports"
    logs_dir = reports_dir / "sharded_logs" / args.shard_prefix
    merge_script = root / "scripts" / "merge_tau2_shards.py"
    summary_script = root / "scripts" / "summarize_tau2_results.py"

    if not tau2_root.exists():
        raise SystemExit(f"tau2-bench checkout not found: {tau2_root}")

    task_ids = parse_task_ids(args.task_ids)
    if not task_ids:
        raise SystemExit("--task-ids is required, e.g. --task-ids 1,2,7")

    uv = find_uv()
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = logs_dir / "manifest.csv"
    if args.force and manifest_path.exists():
        manifest_path.unlink()

    os.environ["PYTHONUTF8"] = "1"

    print("Running timeout-safe tau2 airline shards...")
    print(f"  agent:   {args.agent_llm}")
    print(f"  user:    {args.user_llm}")
    print(f"  tasks:   {','.join(task_ids)}")
    print(f"  trials:  {args.num_trials}")
    print(f"  timeout: {args.timeout_seconds}s per shard")
    print(f"  prefix:  {args.shard_prefix}")
    print(f"  logs:    {logs_dir}")

    total = len(task_ids) * args.num_trials
    index = 0
    for trial in range(args.num_trials):
        for task_id in task_ids:
            index += 1
            cmd, shard_seed, save_to = build_tau2_command(args, uv, task_id, trial)
            stdout_path = logs_dir / f"{save_to}.stdout.log"
            stderr_path = logs_dir / f"{save_to}.stderr.log"

            if not args.force and completed_shard(simulations_root, save_to):
                print(f"[{index}/{total}] skip completed task={task_id} trial={trial} save={save_to}")
                append_manifest(
                    manifest_path,
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "task_id": task_id,
                        "trial": trial,
                        "seed": shard_seed,
                        "status": "skipped",
                        "exit_code": "",
                        "duration_seconds": "0.0",
                        "save_to": save_to,
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                    },
                )
                continue

            print(f"[{index}/{total}] run task={task_id} trial={trial} seed={shard_seed} save={save_to}")
            if args.dry_run:
                print(" ".join(cmd))
                continue

            start = time.monotonic()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                proc = subprocess.Popen(
                    cmd,
                    cwd=tau2_root,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    **popen_kwargs(),
                )
                try:
                    exit_code = proc.wait(timeout=args.timeout_seconds)
                    status = "ok" if exit_code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    stop_process_tree(proc)
                    exit_code = ""
                    status = "timeout"

            duration = time.monotonic() - start
            append_manifest(
                manifest_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "task_id": task_id,
                    "trial": trial,
                    "seed": shard_seed,
                    "status": status,
                    "exit_code": exit_code,
                    "duration_seconds": f"{duration:.1f}",
                    "save_to": save_to,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            )
            print(f"[{index}/{total}] {status} task={task_id} trial={trial} duration={duration:.1f}s")

    if args.dry_run or args.no_merge:
        return

    merged_save_to = args.merged_save_to or f"{args.shard_prefix}_merged"
    merged_out = simulations_root / merged_save_to
    merge_cmd = [
        sys.executable,
        str(merge_script),
        "--shards-root",
        str(simulations_root),
        "--prefix",
        args.shard_prefix,
        "--out",
        str(merged_out),
        "--num-trials",
        str(args.num_trials),
        "--task-ids",
        ",".join(task_ids),
        "--rewrite-trial-from-name",
    ]
    subprocess.run(merge_cmd, cwd=root, check=True)

    summary_out = reports_dir / f"{merged_save_to}.summary.json"
    subprocess.run(
        [sys.executable, str(summary_script), str(merged_out), "--out", str(summary_out)],
        cwd=root,
        check=True,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tau2 airline evals as timeout-safe shards.")
    parser.add_argument("--agent-llm", default="anthropic/glm-5.1")
    parser.add_argument("--user-llm", default="anthropic/glm-5.1")
    parser.add_argument("--agent-llm-args", default="")
    parser.add_argument("--user-llm-args", default="")
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--shard-prefix", default="airline_sharded")
    parser.add_argument("--merged-save-to", default="")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose-logs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    run_shards(make_parser().parse_args())


if __name__ == "__main__":
    main()
