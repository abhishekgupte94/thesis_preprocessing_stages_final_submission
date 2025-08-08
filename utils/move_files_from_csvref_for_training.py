import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, Set, List, Tuple

def move_files_from_csv_by_filename(
    csv_file: str,
    column_name: str = "file",
    source_root: str = "/Users/abhishekgupte_macbookpro/Downloads/LAV-DF",
    destination_dir: str = "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean_preprocessing/files/processed/lip_videos",
    move: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Read a CSV with a column containing filenames (e.g., 'file' -> 'abc123.mp4'),
    search for those filenames in all subdirectories directly under source_root,
    and move/copy them to destination_dir.

    Creates a second CSV next to the original called
    '<original_name>_unmoved_due_to_error.csv' containing the rows that failed.
    """

    csv_path = Path(csv_file).resolve()
    csv_parent = csv_path.parent
    output_dir = Path(destination_dir).resolve()
    src_root = Path(source_root).resolve()

    # --- Load CSV ---
    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in {csv_file}")

    # Normalize filenames (strip spaces)
    df[column_name] = df[column_name].astype(str).str.strip()

    # --- Build an index of all files under the three subdirs (recursively) ---
    # If more than 3 subdirs exist, it still works; it scans all immediate dirs under root.
    name_to_path: Dict[str, Path] = {}
    dup_names: Dict[str, List[Path]] = {}

    for subdir in src_root.iterdir():
        if subdir.is_dir():
            for p in subdir.rglob("*"):
                if p.is_file():
                    fname = p.name
                    if fname in name_to_path:
                        # Track duplicates; we'll keep the first and record the conflict
                        dup_names.setdefault(fname, [name_to_path[fname]]).append(p)
                    else:
                        name_to_path[fname] = p.resolve()

    # --- Ensure destination exists ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Process rows & track failures ---
    failed_indices: Set[int] = set()
    failure_reasons: Dict[int, str] = {}

    moved_count = 0
    copied_count = 0
    not_found_count = 0
    error_count = 0
    skipped_exists_count = 0
    duplicate_hint_count = 0

    for idx, row in df.iterrows():
        filename = row[column_name]
        if not filename:
            failed_indices.add(idx)
            failure_reasons[idx] = "empty_filename"
            error_count += 1
            continue

        # Duplicate hint (won't fail by itself, but we record it)
        if filename in dup_names:
            duplicate_hint_count += 1

        # Locate source
        src = name_to_path.get(filename)
        if src is None:
            failed_indices.add(idx)
            failure_reasons[idx] = "not_found_in_source_root"
            not_found_count += 1
            continue

        # Destination path
        dst = output_dir / filename

        # Collision handling
        if dst.exists():
            if overwrite:
                try:
                    if move:
                        # Remove existing then move
                        dst.unlink()
                        shutil.move(str(src), str(dst))
                        moved_count += 1
                    else:
                        shutil.copy2(str(src), str(dst))
                        copied_count += 1
                except Exception as e:
                    failed_indices.add(idx)
                    failure_reasons[idx] = f"overwrite_failed: {type(e).__name__}: {e}"
                    error_count += 1
            else:
                # Do not overwrite; treat as failure so it appears in the report
                failed_indices.add(idx)
                failure_reasons[idx] = "destination_exists"
                skipped_exists_count += 1
            continue

        # Perform move/copy
        try:
            if move:
                shutil.move(str(src), str(dst))
                moved_count += 1
            else:
                shutil.copy2(str(src), str(dst))
                copied_count += 1
        except Exception as e:
            failed_indices.add(idx)
            failure_reasons[idx] = f"{type(e).__name__}: {e}"
            error_count += 1

    # --- Write failures CSV (same columns/data) next to the original CSV ---
    if failed_indices:
        failed_df = df.loc[sorted(failed_indices)].copy()
        # Optionally include a reason column for debugging (comment out if you truly only want the original columns)
        failed_df["error_reason"] = failed_df.index.map(lambda i: failure_reasons.get(i, "unknown_error"))

        out_csv = csv_parent / f"{csv_path.stem}_unmoved_due_to_error.csv"
        failed_df.to_csv(out_csv, index=False)
        print(f"\n⚠️ Unmoved rows saved to: {out_csv}")
    else:
        print("\n✅ All files moved/copied successfully. No failure CSV created.")

    # --- Summary ---
    total = len(df)
    print("\n=== Summary ===")
    print(f"CSV rows:                    {total}")
    print(f"Moved:                       {moved_count}")
    print(f"Copied:                      {copied_count}")
    print(f"Not found in source:         {not_found_count}")
    print(f"Destination already exists:  {skipped_exists_count} (overwrite={overwrite})")
    print(f"Other errors:                {error_count}")
    if duplicate_hint_count:
        print(f"Duplicate filename hints:    {duplicate_hint_count}  (same filename appears under multiple subdirs)")

if __name__ == "__main__":
    # ---- Edit these paths before running ----
    # move_files_from_csv_by_filename(
    #     csv_file="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean_preprocessing/files/csv_files/processed/video/sample_real_70_percent_half1.csv",                 # path to your CSV
    #     column_name="file",                       # column that contains the filenames
    #     source_root="/Users/abhishekgupte_macbookpro/Downloads/LAV-DF",
    #     destination_dir="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean_preprocessing/datasets/ssl_train/sample_real_70_percent_half1",
    #     move=True,                                # set False to copy instead
    #     overwrite=False,                          # set True to overwrite same-named files at destination
    # )
    # move_files_from_csv_by_filename(
    #     csv_file="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean_preprocessing/files/csv_files/processed/video/sample_real_70_percent_half2.csv",                 # path to your CSV
    #     column_name="file",                       # column that contains the filenames
    #     source_root="/Users/abhishekgupte_macbookpro/Downloads/LAV-DF",
    #     destination_dir="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean_preprocessing/datasets/ssl_train/sample_real_70_percent_half2",
    #     move=True,                                # set False to copy instead
    #     overwrite=False,                          # set True to overwrite same-named files at destination
    # )
    move_files_from_csv_by_filename(
        csv_file="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean_preprocessing/files/csv_files/processed/video/holdout_30_percent_for_training.csv",                 # path to your CSV
        column_name="file",                       # column that contains the filenames
        source_root="/Users/abhishekgupte_macbookpro/Downloads/LAV-DF",
        destination_dir="/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean_preprocessing/datasets/ssl_train/holdout_30_percent_for_training",
        move=True,                                # set False to copy instead
        overwrite=False,                          # set True to overwrite same-named files at destination
    )