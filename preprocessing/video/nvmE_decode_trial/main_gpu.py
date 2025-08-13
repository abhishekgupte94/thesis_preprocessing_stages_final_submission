# main_gpu.py
import os
import sys
import subprocess
from pathlib import Path
import argparse

# --- Bootstrap project root ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.data_loader_ART import (
    get_project_root,
    convert_paths,
    create_file_paths,
)
from utils_csv_sharding import split_rows_round_robin

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Video preprocessing and training setup")
    parser.add_argument('--csv_name', type=str, required=True,
                        help="Name of the CSV file (without '.csv') to use for training or preprocessing.")
    parser.add_argument('--n_gpus', type=int, default=8, help="Number of GPUs to shard over")
    parser.add_argument('--batch_frames', type=int, default=64)
    parser.add_argument('--mouth_margin', type=float, default=0.35)
    parser.add_argument('--size', type=int, nargs=2, default=[224,224])
    parser.add_argument('--path_column', default="file")  # worker expects this
    parser.add_argument('--precomputed_shards_dir', type=str, default=None,
                        help="If set, use existing gpu*/shard.csv here instead of splitting.")
    args = parser.parse_args()

    # --- Prepare paths ---
    csv_file = f"{args.csv_name}.csv"
    csv_name = args.csv_name
    project_root_dir = get_project_root("thesis_preprocessing_stages_final_submission")
    csv_path, _, video_postprocess_dir = convert_paths(csv_file=csv_file, csv_name=csv_name)
    print(f"[main_gpu] Output directory: {video_postprocess_dir}")
    ensure_dir(video_postprocess_dir)

    logs_dir  = Path(video_postprocess_dir) / "logs_preproc"
    ensure_dir(logs_dir)

    # --- Load shards ---
    if args.precomputed_shards_dir:
        shards_root = Path(args.precomputed_shards_dir)
        shard_paths = sorted(shards_root.glob("gpu*/shard.csv"))
        if not shard_paths:
            raise ValueError(f"No shard.csv files found under {shards_root}/gpu*/")
        print(f"[main_gpu] Using {len(shard_paths)} precomputed shards from {shards_root}")
    else:
        # Expand full original paths using your create_file_paths()
        lips_only_paths, original_paths, labels = create_file_paths(
            project_dir_curr=project_root_dir,
            csv_file=csv_file,
            csv_name=csv_name,
            check_original_files=False,
            check_lips_files=False,
            abort_on_missing=False,
            verbose=False
        )

        # Build in-memory rows for sharding
        rows = [{"file": p, "label": lbl} for p, lbl in zip(original_paths, labels)]

        shards_dir = logs_dir / "shards"
        ensure_dir(shards_dir)
        shard_paths = split_rows_round_robin(
            rows=rows,
            headers=["file", "label"],
            out_dir=shards_dir,
            n=args.n_gpus,
            basename="shard"
        )
        print(f"[main_gpu] Created {len(shard_paths)} shard CSVs in {shards_dir}")

    # --- Launch one worker per GPU ---
    procs = []
    for i, shard_csv in enumerate(shard_paths):
        # Determine GPU index
        if args.precomputed_shards_dir:
            try:
                i = int(Path(shard_csv).parent.name.replace("gpu", ""))
            except ValueError:
                pass  # fallback to enumerate index

        gpu_out_dir = Path(video_postprocess_dir) / f"gpu{i}"
        gpu_logs    = logs_dir / f"gpu{i}"
        ensure_dir(gpu_out_dir)
        ensure_dir(gpu_logs)

        success_log = gpu_logs / "success.csv"
        error_log   = gpu_logs / "errors.csv"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)  # pin worker to GPU i

        cmd = [
            sys.executable, "-u", str(ROOT / "main" / "worker_from_csv.py"),
            "--csv", str(shard_csv),
            "--out_dir", str(gpu_out_dir),
            "--device", "cuda:0",                         # maps to visible GPU
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

    # --- Wait for all workers ---
    rc = 0
    for p, out, err in procs:
        p.wait()
        out.close()
        err.close()
        if p.returncode != 0:
            rc = p.returncode

    raise SystemExit(rc)

if __name__ == "__main__":
    main()
