import os
import sys
from pathlib import Path
import argparse
import pandas as pd
# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Project imports
from data_loader.data_loader_ART import (
    get_project_root,
    create_file_paths,
    convert_paths,
)


def ensure_dir(p: Path):
    """Create directory if it doesn't exist"""
    p.mkdir(parents=True, exist_ok=True)


def create_file_shards(video_paths, out_dir, n_gpus, use_symlinks=True):
    """
    Distribute video files across GPU directories using symlinks or copies.

    Args:
        video_paths: List of absolute paths to video files
        out_dir: Base output directory
        n_gpus: Number of GPUs to shard across
        use_symlinks: If True, create symlinks; if False, copy files

    Returns:
        List of GPU directories containing the sharded files
    """
    base = Path(out_dir)
    gpu_dirs = []

    # Create GPU directories
    for i in range(n_gpus):
        gpu_dir = base / f"gpu{i}"
        ensure_dir(gpu_dir)
        gpu_dirs.append(gpu_dir)

    # Distribute files round-robin
    for idx, video_path in enumerate(video_paths):
        gpu_idx = idx % n_gpus
        gpu_dir = gpu_dirs[gpu_idx]

        src_path = Path(video_path)
        if not src_path.exists():
            print(f"[WARNING] Video not found: {video_path}")
            continue

        # Create destination path maintaining filename
        dst_path = gpu_dir / src_path.name

        try:
            if use_symlinks:
                # Remove existing symlink if present
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                # Create symlink to original file
                dst_path.symlink_to(src_path.absolute())
            else:
                # Copy file (more disk space but more portable)
                import shutil
                shutil.copy2(src_path, dst_path)

        except Exception as e:
            print(f"[ERROR] Failed to process {video_path}: {e}")

    # Print summary
    for i, gpu_dir in enumerate(gpu_dirs):
        video_count = len(list(gpu_dir.glob("*.mp4"))) + len(list(gpu_dir.glob("*.avi")))
        print(f"[OK] GPU {i}: {gpu_dir} - {video_count} videos")

    return gpu_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Pre-create file-based shards for multi-GPU processing"
    )
    parser.add_argument("--csv_name", required=True,
                        help="Name of the CSV file (without '.csv') containing video paths")
    parser.add_argument("--n_gpus", type=int, default=8,
                        help="Number of GPUs to shard across")
    parser.add_argument("--use_copy", action="store_true",
                        help="Copy files instead of creating symlinks")
    parser.add_argument("--custom_shard_dir", type=str, default=None,
                        help="Override default shard directory location")
    args = parser.parse_args()

    # Get project root and expand paths
    project_root = get_project_root("thesis_preprocessing_stages_final_submission")
    csv_file = f"{args.csv_name}.csv"

    # Import convert_paths to get the standard output directory
    from data_loader.data_loader_ART import convert_paths

    # Get the standard video_postprocess_dir
    csv_path, _, video_postprocess_dir = convert_paths(
        csv_file=csv_file,
        csv_name=args.csv_name
    )

    # Use video_postprocess_dir/shards as default, or custom if specified
    if args.custom_shard_dir:
        shard_base_dir = Path(args.custom_shard_dir)
    else:
        shard_base_dir = Path(video_postprocess_dir) / "shards"

    print(f"[INFO] Loading video paths from {csv_file}")
    print(f"[INFO] Shards will be created in: {shard_base_dir}")

    # Expand to absolute file paths via project utility
    lips_paths, original_paths, labels = create_file_paths(
        project_dir_curr=project_root,
        csv_file=csv_file,
        csv_name=args.csv_name,
        check_original_files=False,
        check_lips_files=False,
        abort_on_missing=False,
        verbose=False,
    )

    print(f"[INFO] Found {len(original_paths)} videos to process")

    # Create file-based shards
    use_symlinks = not args.use_copy
    gpu_dirs = create_file_shards(
        video_paths=original_paths,
        out_dir=shard_base_dir,
        n_gpus=args.n_gpus,
        use_symlinks=use_symlinks
    )

    print(f"\n[SUCCESS] Created {len(gpu_dirs)} shards in {shard_base_dir}")
    print(f"[INFO] Sharding method: {'symlinks' if use_symlinks else 'file copies'}")


if __name__ == "__main__":
    main()