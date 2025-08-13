# launch_shards_csv.py
import os, csv, math, subprocess
from pathlib import Path
import argparse

def split_csv(in_csv: Path, out_dir: Path, n: int, path_column: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # load rows
    with open(in_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input CSV has no rows.")
    if path_column not in rows[0]:
        raise ValueError(f"CSV missing required column '{path_column}'")

    # simple round-robin split
    shards = [[] for _ in range(n)]
    for i, r in enumerate(rows):
        shards[i % n].append(r)

    shard_paths = []
    for i, shard in enumerate(shards):
        sp = out_dir / f"shard_{i}.csv"
        with open(sp, "w", newline="") as s:
            w = csv.DictWriter(s, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(shard)
        shard_paths.append(sp)
    return shard_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Main CSV with column 'file' (or override with --path_column)")
    ap.add_argument("--out_dir", required=True, help="Base output directory (each GPU gets a subfolder)")
    ap.add_argument("--n_gpus", type=int, default=8)
    ap.add_argument("--batch_frames", type=int, default=64)
    ap.add_argument("--mouth_margin", type=float, default=0.35)
    ap.add_argument("--size", type=int, nargs=2, default=[224,224])
    ap.add_argument("--path_column", default="file")
    ap.add_argument("--logs_dir", default="logs_preproc")
    args = ap.parse_args()

    in_csv = Path(args.csv)
    shards_dir = Path(args.logs_dir) / "shards"
    shards = split_csv(in_csv, shards_dir, args.n_gpus, args.path_column)

    # launch one worker per GPU with CUDA_VISIBLE_DEVICES=i
    procs = []
    for i, shard_csv in enumerate(shards):
        gpu_out_dir = Path(args.out_dir) / f"gpu{i}"
        gpu_logs = Path(args.logs_dir) / f"gpu{i}"
        gpu_logs.mkdir(parents=True, exist_ok=True)
        success_log = gpu_logs / "success.csv"
        error_log   = gpu_logs / "errors.csv"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)

        cmd = [
            "python", "-u", "worker_from_csv.py",
            "--csv", str(shard_csv),
            "--out_dir", str(gpu_out_dir),
            "--device", "cuda:0",
            "--batch_frames", str(args.batch_frames),
            "--mouth_margin", str(args.mouth_margin),
            "--size", str(args.size[0]), str(args.size[1]),
            "--success_log", str(success_log),
            "--error_log", str(error_log),
            "--path_column", args.path_column,
        ]
        stdout = open(gpu_logs / "stdout.log", "w")
        stderr = open(gpu_logs / "stderr.log", "w")
        p = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
        procs.append((p, stdout, stderr))

    # wait for workers
    rc = 0
    for p, out, err in procs:
        p.wait()
        out.close(); err.close()
        if p.returncode != 0:
            rc = p.returncode
    raise SystemExit(rc)

if __name__ == "__main__":
    from pathlib import Path
    main()
