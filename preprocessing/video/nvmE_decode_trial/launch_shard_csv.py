import os
import sys
import subprocess
from pathlib import Path
import argparse

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.data_loader_ART import (
    get_project_root,
    convert_paths,
    create_file_paths,
)


def ensure_dir(p: Path):
    """Create directory if it doesn't exist"""
    p.mkdir(parents=True, exist_ok=True)


def create_file_shards_inline(video_paths, shards_dir, n_gpus):
    """
    Quick inline sharding - creates directories with symlinks.
    Returns list of shard directories.
    """
    gpu_dirs = []

    for i in range(n_gpus):
        gpu_dir = shards_dir / f"gpu{i}"
        ensure_dir(gpu_dir)
        gpu_dirs.append(gpu_dir)

    # Distribute files round-robin
    for idx, video_path in enumerate(video_paths):
        gpu_idx = idx % n_gpus
        gpu_dir = gpu_dirs[gpu_idx]

        src_path = Path(video_path)
        if not src_path.exists():
            continue

        dst_path = gpu_dir / src_path.name

        # Create symlink
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        dst_path.symlink_to(src_path.absolute())

    return gpu_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Launch multi-GPU video preprocessing with file-based sharding"
    )
    # CSV and sharding options
    parser.add_argument('--csv_name', type=str, required=True,
                        help="Name of the CSV file (without '.csv') to use for training or preprocessing.")
    parser.add_argument('--n_gpus', type=int, default=8,
                        help="Number of GPUs to use")
    parser.add_argument('--precomputed_shards_dir', type=str, default=None,
                        help="If set, use existing gpu*/ directories here instead of creating new shards")

    # Processing parameters
    parser.add_argument('--batch_frames', type=int, default=64,
                        help="Number of frames to process in each batch")
    parser.add_argument('--mouth_margin', type=float, default=0.35,
                        help="Margin around mouth region")
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224],
                        help="Output size for lip crops")

    # Output options
    parser.add_argument('--output_base_dir', type=str, default=None,
                        help="Base directory for output (defaults to project structure)")

    args = parser.parse_args()

    # Setup paths using project utilities
    csv_file = f"{args.csv_name}.csv"
    csv_name = args.csv_name
    project_root_dir = get_project_root("thesis_preprocessing_stages_final_submission")

    # Use convert_paths to get proper output directory
    csv_path, _, video_postprocess_dir = convert_paths(
        csv_file=csv_file,
        csv_name=csv_name
    )

    # Override output directory if specified
    if args.output_base_dir:
        video_postprocess_dir = Path(args.output_base_dir)
    else:
        video_postprocess_dir = Path(video_postprocess_dir)

    print(f"[INFO] Output directory: {video_postprocess_dir}")
    ensure_dir(video_postprocess_dir)

    logs_dir = video_postprocess_dir / "logs_preproc"
    ensure_dir(logs_dir)

    # Determine shard directories
    if args.precomputed_shards_dir:
        # Use existing shards
        shards_root = Path(args.precomputed_shards_dir)
        shard_dirs = sorted(shards_root.glob("gpu*/"))
        if not shard_dirs:
            raise ValueError(f"No gpu*/ directories found under {shards_root}")
        print(f"[INFO] Using {len(shard_dirs)} precomputed shards from {shards_root}")
    else:
        # Create new shards in the standard location
        print(f"[INFO] Creating file-based shards...")

        # Load video paths using project utilities
        lips_only_paths, original_paths, labels = create_file_paths(
            project_dir_curr=project_root_dir,
            csv_file=csv_file,
            csv_name=csv_name,
            check_original_files=False,
            check_lips_files=False,
            abort_on_missing=False,
            verbose=False
        )

        print(f"[INFO] Found {len(original_paths)} videos to process")

        # Create shards in video_postprocess_dir/shards
        shards_dir = video_postprocess_dir / "shards"
        ensure_dir(shards_dir)
        shard_dirs = create_file_shards_inline(
            video_paths=original_paths,
            shards_dir=shards_dir,
            n_gpus=args.n_gpus
        )
        print(f"[INFO] Created {len(shard_dirs)} shard directories in {shards_dir}")

    # Launch one worker per GPU with separate output directories
    procs = []
    for i, shard_dir in enumerate(shard_dirs):
        # Extract GPU index from directory name
        if args.precomputed_shards_dir:
            try:
                i = int(shard_dir.name.replace("gpu", ""))
            except ValueError:
                pass  # fallback to enumerate index

        # Create separate output directory for each GPU
        gpu_output_base = video_postprocess_dir / f"output_gpu{i}"
        gpu_out_dir = gpu_output_base / "processed_videos"
        gpu_logs = gpu_output_base / "logs"

        ensure_dir(gpu_out_dir)
        ensure_dir(gpu_logs)

        success_log = gpu_logs / "success.csv"
        error_log = gpu_logs / "errors.csv"

        print(f"\n[GPU {i}] Configuration:")
        print(f"  - Input shard: {shard_dir}")
        print(f"  - Output dir: {gpu_out_dir}")
        print(f"  - Logs dir: {gpu_logs}")

        # Set CUDA_VISIBLE_DEVICES to isolate GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)

        # Launch worker process
        cmd = [
            sys.executable, "-u", str(ROOT / "main" / "worker_gpu.py"),
            "--input_dir", str(shard_dir),
            "--out_dir", str(gpu_out_dir),
            "--device", "cuda:0",  # Always cuda:0 since we use CUDA_VISIBLE_DEVICES
            "--batch_frames", str(args.batch_frames),
            "--mouth_margin", str(args.mouth_margin),
            "--size", str(args.size[0]), str(args.size[1]),
            "--success_log", str(success_log),
            "--error_log", str(error_log),
        ]

        stdout_log = open(gpu_logs / "stdout.log", "w")
        stderr_log = open(gpu_logs / "stderr.log", "w")

        print(f"[INFO] Launching worker for GPU {i} processing {shard_dir}")
        p = subprocess.Popen(cmd, env=env, stdout=stdout_log, stderr=stderr_log)
        procs.append((p, stdout_log, stderr_log, i))

    # Wait for all workers to complete
    print(f"\n[INFO] Waiting for {len(procs)} workers to complete...")

    return_code = 0
    for p, out, err, gpu_idx in procs:
        p.wait()
        out.close()
        err.close()

        if p.returncode != 0:
            print(f"[ERROR] Worker for GPU {gpu_idx} failed with return code {p.returncode}")
            return_code = p.returncode
        else:
            print(f"[OK] Worker for GPU {gpu_idx} completed successfully")

    if return_code == 0:
        print("\n[SUCCESS] All workers completed successfully!")
    else:
        print(f"\n[ERROR] Some workers failed. Check logs in {logs_dir}")

    raise SystemExit(return_code)


if __name__ == "__main__":
    main()