# Ensure the root of the project is in sys.path
import os
import sys
# import os
from pathlib import Path
import sys

# /.../project/main/main_gpu.py -> parent[1] == /.../project
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# from codecarbon import EmissionsTracker

from data_loader.data_loader_ART import get_project_root,convert_paths,create_file_paths,preprocess_videos_before_training

# import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse


# Sanity check - file paths
def main():
    # 1. Argument parser
    parser = argparse.ArgumentParser(description="Video preprocessing and training setup")
    parser.add_argument('--csv_name', type=str, required=True,
                        help="Name of the CSV file (without '.csv') to use for training or preprocessing.")
    args = parser.parse_args()

    # 2. Auto-append .csv to the csv name
    csv_file = f"{args.csv_name}.csv"

    project_root_dir = get_project_root()
    csv_path,_,video_postprocess_dir = convert_paths()
    if csv_file:
        _,video_paths,_ = create_file_paths(project_root_dir,csv_name = csv_file)
        print(video_paths[1:10])
        preprocess_videos_before_training(csv_name = csv_file,output_dir=video_postprocess_dir)
    else:
        print("The csv name does not exist!")


if __name__ == "__main__":
    main()

# print(str(project_root))
# lip_paths, original_paths, labels = create_file_paths(project_dir_curr= project_root,csv_name = "possible_training_sample.csv")
# lip_paths_sample = lip_paths[1:5]
# original_paths_sample = original_paths[1:5]
#
# print(lip_paths_sample,original_paths_sample)
# main()