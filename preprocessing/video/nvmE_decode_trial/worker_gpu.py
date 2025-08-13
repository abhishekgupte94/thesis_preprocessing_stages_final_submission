# worker_gpu.py
import os, csv
from pathlib import Path
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
from fast_preproc_decord_mouth import process_video_gpu_lips



def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to a shard CSV containing a 'file' column with absolute video paths")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_frames", type=int, default=64)
    ap.add_argument("--mouth_margin", type=float, default=0.35)
    ap.add_argument("--size", type=int, nargs=2, default=[224,224])
    ap.add_argument("--success_log", required=True)
    ap.add_argument("--error_log", required=True)
    ap.add_argument("--path_column", default="file")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    success_log = Path(args.success_log); error_log = Path(args.error_log)
    ensure_dir(success_log.parent); ensure_dir(error_log.parent)

    # open logs
    with open(success_log, "w", newline="") as fs, open(error_log, "w", newline="") as fe:
        succ_writer = csv.DictWriter(fs, fieldnames=[
            "video","out","elapsed_sec","frames_total","frames_written","fps_effective","gpu","status"
        ])
        err_writer = csv.DictWriter(fe, fieldnames=[
            "video","error","gpu","status"
        ])
        succ_writer.writeheader()
        err_writer.writeheader()

        # read shard csv
        with open(args.csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            if args.path_column not in reader.fieldnames:
                raise ValueError(f"CSV missing required column '{args.path_column}'")
            for row in reader:
                video = row[args.path_column]
                if not video:
                    err_writer.writerow({"video": "", "error": "empty path", "gpu": args.device, "status": "skipped"})
                    continue
                try:
                    vid_path = Path(video)
                    if not vid_path.is_file():
                        raise FileNotFoundError(f"not found: {video}")
                    out_path = str(out_dir / (vid_path.stem + "_lips.mp4"))

                    stats = process_video_gpu_lips(
                        video_path=video,
                        out_path=out_path,
                        device=args.device,
                        out_size=tuple(args.size),
                        batch_frames=args.batch_frames,
                        mouth_margin=args.mouth_margin
                    )
                    stats["gpu"] = args.device
                    stats["status"] = "ok"
                    succ_writer.writerow(stats)
                    fs.flush()
                except Exception as e:
                    err_writer.writerow({
                        "video": video,
                        "error": str(e),
                        "gpu": args.device,
                        "status": "failed"
                    })
                    fe.flush()

if __name__ == "__main__":
    main()
