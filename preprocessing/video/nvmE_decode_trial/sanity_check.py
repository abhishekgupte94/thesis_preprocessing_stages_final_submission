#!/usr/bin/env python3
"""
Sanity check for single GPU with automatic shard detection.
Uses the same project structure as prepare_shards.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import pandas as pd
import argparse

# --- Make sure we add the directory that directly contains `data_loader/` ---
here = Path(__file__).resolve()
ROOT = None
for parent in [here.parent, *here.parents]:
    if (parent / "data_loader").is_dir():
        ROOT = parent
        break

if ROOT is None:
    raise ImportError(
        f"Could not find a parent directory containing 'data_loader/' starting from {here}"
    )

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.data_loader_ART import (
    get_project_root,
    convert_paths,
)


def run_single_gpu_test(shard_dir, gpu_id, test_videos=5):
    """
    Run a quick test on a single GPU with one shard.
    """

    print("=" * 60)
    print(f"SINGLE GPU SANITY CHECK")
    print(f"Shard: {shard_dir}")
    print(f"GPU: {gpu_id}")
    print("=" * 60)

    # Verify shard exists
    shard_path = Path(shard_dir)
    if not shard_path.exists():
        print(f"[ERROR] Shard directory not found: {shard_dir}")
        return False

    # Count videos
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    video_files = []
    for ext in video_extensions:
        video_files.extend(shard_path.glob(f"*{ext}"))

    total_videos = len(video_files)
    print(f"\n[INFO] Found {total_videos} videos in shard")

    if total_videos == 0:
        print("[ERROR] No videos found in shard!")
        return False

    # Create test subset if requested
    if test_videos and test_videos < total_videos:
        print(f"[INFO] Testing with first {test_videos} videos only")
        test_dir = Path("./sanity_test_subset")
        test_dir.mkdir(exist_ok=True)

        # Clean previous test files
        for old_file in test_dir.glob("*"):
            if old_file.is_file() or old_file.is_symlink():
                old_file.unlink()

        # Create symlinks for test videos
        for video in sorted(video_files)[:test_videos]:
            link = test_dir / video.name
            if link.exists():
                link.unlink()
            link.symlink_to(video.absolute())

        input_dir = test_dir
        num_to_process = test_videos
    else:
        input_dir = shard_path
        num_to_process = total_videos
        print(f"[INFO] Testing with all {total_videos} videos")

    # Get video_postprocess_dir to match the structure
    csv_name = shard_path.parent.parent.name.replace("video_postprocess_", "")

    # Setup output directories - use similar structure to launch_shard.py
    output_base = shard_path.parent.parent  # video_postprocess_dir
    gpu_output_base = output_base / f"output_gpu{gpu_id}_sanity"
    output_dir = gpu_output_base / "processed_videos"
    logs_dir = gpu_output_base / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Output structure:")
    print(f"  - Base: {output_base}")
    print(f"  - Output: {output_dir}")
    print(f"  - Logs: {logs_dir}")

    # Find worker script
    worker_paths = [
        ROOT / "preprocessing" / "video"/ "nvmE_decode_trial" / "worker_gpu.py",
        ROOT / "main" / "worker_gpu.py",
        Path("worker_gpu.py"),
        Path("main/worker_gpu.py"),
    ]

    worker_script = None
    for path in worker_paths:
        if path.exists():
            worker_script = path
            break

    if not worker_script:
        print("[ERROR] Cannot find worker_gpu.py")
        print("Searched in:", [str(p) for p in worker_paths])
        return False

    print(f"[INFO] Using worker script: {worker_script}")

    # Prepare command
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "-u", str(worker_script),
        "--input_dir", str(input_dir),
        "--out_dir", str(output_dir),
        "--device", "cuda:0",
        "--batch_frames", "64",
        "--mouth_margin", "0.35",
        "--size", "224", "224",
        "--success_log", str(logs_dir / "success.csv"),
        "--error_log", str(logs_dir / "errors.csv"),
    ]

    print(f"\n[INFO] Starting processing...")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"\n{'=' * 60}\n")

    # Run worker with real-time output
    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output
        for line in process.stdout:
            print(line.rstrip())

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode != 0:
            print(f"\n[ERROR] Worker failed with return code {process.returncode}")
            return False

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n[ERROR] Failed to run worker: {e}")
        return False

    print(f"\n{'=' * 60}")
    print(f"[INFO] Processing completed in {elapsed:.1f} seconds")
    print(f"{'=' * 60}\n")

    # Analyze results
    success_log = logs_dir / "success.csv"
    error_log = logs_dir / "errors.csv"

    if success_log.exists():
        df = pd.read_csv(success_log)
        num_success = len(df)

        if num_success > 0:
            avg_fps = df['fps_effective'].mean()
            total_frames = df['frames_total'].sum()
            total_written = df['frames_written'].sum()
            avg_time = df['elapsed_sec'].mean()

            print("[RESULTS SUMMARY]")
            print(f"- Videos processed: {num_success}/{num_to_process}")
            print(f"- Average FPS: {avg_fps:.1f}")
            print(f"- Average time per video: {avg_time:.2f} seconds")
            print(f"- Total frames: {total_frames:,}")
            print(f"- Frames with lips detected: {total_written:,} ({total_written / total_frames * 100:.1f}%)")
            print(f"- Processing rate: {num_success / elapsed * 60:.1f} videos/minute")

            # Show sample results
            print(f"\n[SAMPLE RESULTS]")
            print(df[['video', 'frames_total', 'frames_written', 'fps_effective', 'elapsed_sec']].head())

    if error_log.exists() and Path(error_log).stat().st_size > 0:
        error_df = pd.read_csv(error_log)
        print(f"\n[ERRORS] {len(error_df)} videos failed:")
        print(error_df.head())

    # Check output files
    output_files = list(output_dir.glob("*_lips.mp4"))
    if output_files:
        print(f"\n[OUTPUT FILES] Created {len(output_files)} lip videos")
        total_size = sum(f.stat().st_size for f in output_files) / 1e6
        print(f"- Total output size: {total_size:.1f} MB")
        print(f"- Average size: {total_size / len(output_files):.1f} MB per video")

        # Show first few
        print("\nSample outputs:")
        for f in sorted(output_files)[:5]:
            size_mb = f.stat().st_size / 1e6
            print(f"  - {f.name} ({size_mb:.1f} MB)")

    # Performance projection
    if num_success > 0 and test_videos and test_videos < total_videos:
        projected_time = (elapsed / num_success) * total_videos
        print(f"\n[PROJECTION]")
        print(f"- Estimated time for all {total_videos} videos: {projected_time / 60:.1f} minutes")
        print(f"- Estimated throughput: {total_videos / (projected_time / 60):.0f} videos/minute")

    # Cleanup test directory if used
    if test_videos and test_videos < total_videos:
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check for single GPU with automatic shard detection"
    )
    parser.add_argument("--csv_name", required=True,
                        help="Name of the CSV file (without '.csv')")
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU ID to test (e.g., 0, 1, 2...)")
    parser.add_argument("--test_videos", type=int, default=5,
                        help="Number of videos to test (default: 5, use 0 for all)")
    parser.add_argument("--custom_shard_dir", type=str, default=None,
                        help="Override automatic shard detection with custom path")

    args = parser.parse_args()

    if args.custom_shard_dir:
        # Use custom shard directory
        shard_dir = Path(args.custom_shard_dir)
    else:
        # Auto-detect shard directory using project structure
        csv_file = f"{args.csv_name}.csv"
        csv_name = args.csv_name

        # Get project root
        project_root_dir = get_project_root("thesis_preprocessing_stages_final_submission")

        # Use convert_paths to get the standard output directory
        csv_path, _, video_postprocess_dir = convert_paths(
            csv_file=csv_file,
            csv_name=csv_name
        )

        # Build shard path
        shard_dir = Path(video_postprocess_dir) / "shards" / f"gpu{args.gpu}"

        print(f"[INFO] Auto-detected shard directory: {shard_dir}")

    if not shard_dir.exists():
        print(f"[ERROR] Shard directory not found: {shard_dir}")
        print("\nHint: Make sure you've run prepare_shards.py first:")
        print(f"  python prepare_shards.py --csv_name {args.csv_name} --n_gpus 8")
        return 1

    # Handle 0 as "all videos"
    test_videos = args.test_videos if args.test_videos > 0 else None

    success = run_single_gpu_test(
        shard_dir=shard_dir,
        gpu_id=args.gpu,
        test_videos=test_videos
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())