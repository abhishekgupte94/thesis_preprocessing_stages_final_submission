import os
import sys
from pathlib import Path

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

import argparse
import pandas as pd
from fast_preproc_decord_for_llps import process_video_gpu_lips


def ensure_dir(p: Path):
    """Create directory if it doesn't exist"""
    p.mkdir(parents=True, exist_ok=True)


def get_video_files(input_dir):
    """Get all video files in the input directory"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    video_files = []

    for file_path in Path(input_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)

    return sorted(video_files)


def main():
    parser = argparse.ArgumentParser(
        description="GPU worker for processing videos in a directory"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing video files (or symlinks) to process")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for processed videos")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device to use")
    parser.add_argument("--batch_frames", type=int, default=64,
                        help="Number of frames to process in each batch")
    parser.add_argument("--mouth_margin", type=float, default=0.35,
                        help="Margin around mouth region")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224],
                        help="Output size for lip crops")
    parser.add_argument("--success_log", required=True,
                        help="Path to success log CSV")
    parser.add_argument("--error_log", required=True,
                        help="Path to error log CSV")
    args = parser.parse_args()

    # Setup directories
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    success_log = Path(args.success_log)
    error_log = Path(args.error_log)
    ensure_dir(success_log.parent)
    ensure_dir(error_log.parent)

    # Get all video files
    video_files = get_video_files(input_dir)

    if not video_files:
        print(f"[WARNING] No video files found in {input_dir}")
        return

    print(f"[INFO] Found {len(video_files)} videos to process in {input_dir}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Using device: {args.device}")

    # Initialize results lists for pandas DataFrames
    success_records = []
    error_records = []

    # Process each video
    for idx, video_path in enumerate(video_files):
        video_name = video_path.name
        video_str = str(video_path)

        print(f"[{idx + 1}/{len(video_files)}] Processing {video_name}...")

        try:
            # Resolve symlink if necessary
            if video_path.is_symlink():
                actual_path = video_path.resolve()
                if not actual_path.exists():
                    raise FileNotFoundError(f"Symlink target not found: {actual_path}")
                video_str = str(actual_path)

            # Output path maintains the original filename
            out_path = out_dir / (video_path.stem + "_lips.mp4")

            # Process video
            stats = process_video_gpu_lips(
                video_path=video_str,
                out_path=str(out_path),
                device=args.device,
                out_size=tuple(args.size),
                batch_frames=args.batch_frames,
                mouth_margin=args.mouth_margin
            )

            # Add additional info to stats
            stats["gpu"] = args.device
            stats["status"] = "ok"

            success_records.append(stats)

            print(f"  ✓ Success: {stats['frames_written']} frames, "
                  f"{stats['fps_effective']} fps")

        except Exception as e:
            error_record = {
                "video": video_str,
                "error": str(e),
                "gpu": args.device,
                "status": "failed"
            }
            error_records.append(error_record)

            print(f"  ✗ Error: {str(e)}")

    # Save results using pandas
    if success_records:
        success_df = pd.DataFrame(success_records)
        success_df.to_csv(success_log, index=False)
        print(f"\n[SUCCESS] Processed {len(success_df)} videos successfully")

    if error_records:
        error_df = pd.DataFrame(error_records)
        error_df.to_csv(error_log, index=False)
        print(f"[ERRORS] Failed to process {len(error_df)} videos")

    print(f"\n[DONE] Processed {len(video_files)} videos")
    print(f"[INFO] Success log: {success_log}")
    print(f"[INFO] Error log: {error_log}")


if __name__ == "__main__":
    main()