import csv
from pathlib import Path
import sys

# Project imports
from data_loader.data_loader_ART import (
    get_project_root,
    create_file_paths,
)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def split_rows_round_robin(rows, n):
    buckets = [[] for _ in range(n)]
    for i, r in enumerate(rows):
        buckets[i % n].append(r)
    return buckets

def main():
    import argparse
    ap = argparse.ArgumentParser("Pre-create round-robin shards into per-GPU folders")
    ap.add_argument("--csv_name", required=True)
    ap.add_argument("--out_dir", required=True, help="Where to write gpu*/shard.csv")
    ap.add_argument("--n_gpus", type=int, default=8)
    ap.add_argument("--path_column", default="file")
    args = ap.parse_args()

    project_root = get_project_root("thesis_preprocessing_stages_final_submission")
    csv_file = f"{args.csv_name}.csv"

    # Expand to absolute file paths via your utility
    lips_paths, original_paths, labels = create_file_paths(
        project_dir_curr=project_root,
        csv_file=csv_file,
        csv_name=args.csv_name,
        check_original_files=False,
        check_lips_files=False,
        abort_on_missing=False,
        verbose=False,
    )

    rows = [{"file": p, "label": lbl} for p, lbl in zip(original_paths, labels)]
    buckets = split_rows_round_robin(rows, args.n_gpus)

    base = Path(args.out_dir)
    for i, bucket in enumerate(buckets):
        gpu_dir = base / f"gpu{i}"
        ensure_dir(gpu_dir)
        shard_csv = gpu_dir / "shard.csv"
        with open(shard_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "label"])
            w.writeheader()
            w.writerows(bucket)
        print(f"[ok] wrote {shard_csv} ({len(bucket)} rows)")

if __name__ == "__main__":
    main()
