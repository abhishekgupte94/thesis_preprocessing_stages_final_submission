#
from torch.utils.data import Dataset

# Importing required modules for video/audio processed and feature extraction

# from preprocessing.video.video_preprocessor_SAVE_FILES_MULTI_GPU  import  parallel_main
# from thesis_main_files.main_files.feature_extraction.art_avdf.art.feature_extractor_ART_Video import SWIN_EXECUTOR as VideoFeatureExtractor
# from thesis_main_files.main_files.processed.art_avdf.art.audio_preprocessorart import AudioPreprocessor
# from thesis_main_files.main_files.processed.art_avdf.art.video_preprocessorart_Fanet_gpu import VideoPreprocessor_FANET
from preprocessing.video.video_preprocessor_reframed_GPUs import parallel_main


from pathlib import Path
import pandas as pd

def get_project_root(project_name):
    """
    Locate the root directory of the project based on script path.

    Args:
        project_name (str, optional): Specific project name to locate.

    Returns:
        Path or None: Root directory if found, else None.
    """
    import os
    current = Path(os.getcwd()).resolve()

    # Look for the known parent folder
    for parent in current.parents:
        if parent.name == "project_combined_repo_clean_preprocessing":
            base_dir = parent.parent
            break
        elif parent.name == "thesis_preprocessing_stages_final_submission":
            return Path(parent.parent)

    else:
        return None

    # if project_name:
        # Return the matching subdirectory if it exists
    if base_dir and project_name:
        target_path = base_dir / project_name
        if target_path.exists() and target_path.is_dir():
            return target_path
        else:
            return None
    # else:
    #     # Fallback search for common project directories
    #     project_names = {"thesis_main_files", "Video-Swin-Transformer", "melodyExtraction_JDC"}
    #     for parent in current.parents:
    #         if parent.name in project_names:
    #             return parent
    return None
def convert_paths(csv_file):
    """
    Prepare all necessary paths for processing and feature extraction.

    Returns:
        Tuple containing all path strings used for video processed and feature extraction.
    """
    # project_dir_curr = Path("/content/project_combined_repo_clean/thesis_main_files")
    project_dir_curr = get_project_root("thesis_preprocessing_stages_final_submission") #subject to change

    # Construct paths used in processing
    csv_path = str(project_dir_curr / "datasets" / "files" / "csv_files" / "processed" / "video" / csv_file)
    video_dir = str(project_dir_curr / "datasets" / "ssl_train")

    # Swin Transformer project-specific paths
    project_dir= get_project_root()
    video_postprocess_dir = project_dir / "files" / "processed" / "lip_videos" /"sample_real_70_percent_half1" #subject to change

    return csv_path, video_dir, video_postprocess_dir
def create_file_paths(project_dir_curr, csv_name=None):
    """
    Generates full paths for video files based on filenames from a CSV file,
    appending '_lips_only' to each filename before the extension.

    Args:
        project_dir_curr (Path or str): Base project directory.
        csv_name (str): Name of CSV file with file listings and labels.

    Returns:
        tuple: lips_only_paths, original_paths, labels
    """
    # project_dir_curr = get_project_root()

    # CSV and video directory paths
    # / Users / abhishekgupte_macbookpro / PycharmProjects / project_combined_repo_clean_preprocessing / files / processed / lip_videos
    csv_path = project_dir_curr / "files" /"csv_files" / "processed" / "video"/ csv_name
    csv_dir =  project_dir_curr / "files" /"csv_files" / "processed" / "video"
    save_dir = project_dir_curr / "files" / "processed" / "lip_videos"

    df = pd.read_csv(csv_path)
    # project_dir_curr = Path("Video-Swin-Transformer")
    original_file_dir = project_dir_curr / "datasets" / "ssl_train"

    # Read CSV and extract file paths


    lips_only_paths = []
    original_paths = []
    for filename in df['file']:
        original_path = Path(filename)
        new_filename = original_path.stem + "_lips_only" + original_path.suffix
        full_lips_only_path = save_dir / new_filename
        full_original_path = original_file_dir / original_path

        lips_only_paths.append(str(full_lips_only_path))
        original_paths.append(str(full_original_path))

    labels = df['label'].tolist()

    return lips_only_paths, original_paths, labels
def preprocess_videos_before_training(csv_name,output_dir, batch_size=32):
    """
    Reads video paths from a CSV file and preprocesses them into a common output directory.

    Args:
        csv_path (str): Path to the CSV file.
        csv_column (str): Column in CSV containing video paths.
        output_dir (str): Directory where lip-only videos will be saved.
        batch_size (int): Number of frames per batch for lip extraction.
    """
    import pandas as pd
    project_dir_curr = get_project_root()
    # csv_name = Path(csv_path).name
    _ ,original_paths,_ = create_file_paths(project_dir_curr,csv_name = csv_name)
    # Step 1: Read CSV
    # df = pd.read_csv(csv_path)
    # if csv_column not in df.columns:
    #     raise ValueError(f"Column '{csv_column}' not found in {csv_path}")
    #
    # video_paths = df[csv_column].tolist()

    # # Step 2: Initialize Preprocessor
    # preproc = VideoPreprocessor_FANET(
    #     batch_size=batch_size,
    #     output_base_dir=output_dir,
    #     device="cuda" # auto-handled per rank
    #     # use_fp16=True
    #
    # )
    # parallel_main()
    parallel_main(original_paths,batch_size,output_dir)
    # Step 3: Preprocess all videos
    # preproc.parallel_main(video_paths)

    print(f"✅ All videos preprocessed and saved to: {output_dir}")

# from pathlib import Path




# def preprocess_videos_before_evaluation(csv_path, csv_column, output_dir, batch_size=128):
#     """
#     Reads video paths from a CSV file and preprocesses them into a common output directory.
#
#     Args:
#         csv_path (str): Path to the CSV file.
#         csv_column (str): Column in CSV containing video paths.
#         output_dir (str): Directory where lip-only videos will be saved.
#         batch_size (int): Number of frames per batch for lip extraction.
#     """
#     import pandas as pd
#     project_dir_curr = get_project_root()
#     csv_name = Path(csv_path).name
#     _,video_paths,_ = create_file_paths(project_dir_curr,csv_name = csv_name)
#     # Step 1: Read CSV
#     # df = pd.read_csv(csv_path)
#     # if csv_column not in df.columns:
#     #     raise ValueError(f"Column '{csv_column}' not found in {csv_path}")
#     #
#     # video_paths = df[csv_column].tolist()
#
#     # Step 2: Initialize Preprocessor
#     # preproc = VideoPreprocessor_FANET(
#     #     batch_size=batch_size,
#     #     output_base_dir=output_dir,
#     #     device="cuda"  # auto-handled per rank
#     #     # use_fp16=True
#     #
#     # )
#
#
#     # Step 3: Preprocess all videos
#     # preproc.parallel_main(video_paths)
#
#     print(f"✅ All videos preprocessed and saved to: {output_dir}")
# import pandas as pd
#
#
#
#
# def create_file_paths_for_inference_ssl(project_dir_curr, csv_name="sampled_combined_data.csv"):
#     """
#     Generates full paths for video files based on filenames from a CSV file
#     for inference evaluation (no lips-only versions).
#
#     Args:
#         project_dir_curr (Path or str): Base project directory.
#         csv_name (str): Name of CSV file with file listings and labels.
#
#     Returns:
#         tuple: original_paths, labels
#     """
#     project_dir_curr = Path(project_dir_curr)
#
#     # CSV and video directory paths
#     csv_path = project_dir_curr / "datasets" / "processed" / "csv_files" / "dfdc" / "inference_data" / csv_name
#     video_dir = project_dir_curr / "datasets" / "processed" / "dfdc" / "eval"
#     df = pd.read_csv(csv_path)
#
#     original_paths = []
#     for filename in df['filename']:
#         full_original_path = video_dir / filename
#         original_paths.append(full_original_path)
#
#     labels = df['label'].tolist()
#
#     return original_paths, labels
#
#
# def convert_paths_for_svm_train_preprocess():
#     """
#     Prepare all necessary paths for SVM training data processing and feature extraction.
#
#     Returns:
#         Tuple containing all path strings used for video processed and feature extraction.
#     """
#     project_dir_curr = get_project_root()
#
#     # Paths for SVM training data
#     csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "final_training_data_svm"/ "training_data_svm_final.csv")
#     video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "checks" / "data_to_preprocess_for_svm")
#
#     # Swin Transformer project-specific paths (unchanged)
#     project_dir_video_swin = get_project_root("Video-Swin-Transformer")
#     video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
#     real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
#     feature_dir_vid = str(project_dir_video_swin)
#
#     return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path
#
#
# def convert_paths_for_svm_val_preprocess():
#     """
#     Prepare all necessary paths for SVM validation data processing and feature extraction.
#
#     Returns:
#         Tuple containing all path strings used for video processed and feature extraction.
#     """
#     project_dir_curr = get_project_root()
#
#     # Paths for SVM validation data
#     csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "lav_df" / "training_data" / "val_data_for_svm.csv")
#     video_dir = str(project_dir_curr / "datasets" / "processed" / "lav_df" / "checks" / "data_to_preprocess_for_svm_val")
#
#     # Swin Transformer project-specific paths (unchanged)
#     project_dir_video_swin = get_project_root("Video-Swin-Transformer")
#     video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
#     real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
#     feature_dir_vid = str(project_dir_video_swin)
#
#     return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path
# def convert_paths_for_inference_ssl_dfdc():
#     """
#     Prepare all necessary paths for Inference Evaluation processed and feature extraction.
#
#     Returns:
#         Tuple containing all path strings used for video processed and feature extraction.
#     """
#     project_dir_curr = get_project_root()
#
#     # Paths for Inference Evaluation Data
#     csv_path = str(project_dir_curr / "datasets" / "processed" / "csv_files" / "dfdc" / "inference_data" / "sampled_combined_data.csv")
#     video_dir = str(project_dir_curr / "datasets" / "processed" / "dfdc" / "eval")  # <-- Assuming videos are here. Adjust if needed.
#
#     # Swin Transformer project-specific paths (unchanged)
#     project_dir_video_swin = get_project_root("Video-Swin-Transformer")
#     video_preprocess_dir = str(project_dir_video_swin / "data" / "train" / "real")
#     real_output_txt_path = str(project_dir_video_swin / "data" / "train" / "real" / "lip_train_text_real.txt")
#     feature_dir_vid = str(project_dir_video_swin)
#
#     return csv_path, video_preprocess_dir, feature_dir_vid, video_dir, real_output_txt_path
#
#
# ###############################################################################
# # COMPONENT + FEATURE EXTRACTION CLASSES (Audio restored via video paths)
# ###############################################################################
#
# # class VideoComponentExtractor:
# #     """
# #     Handles raw video component extraction using the FANET video preprocessor.
# #     """
# #     def extract_video_components(self, video_paths, video_save_dir, output_txt_file, batch_size, video_preprocessor):
# #         try:
# #             # Process video paths using preprocessor
# #             return_paths = video_preprocessor.parallel_main(video_paths)
# #             return return_paths.copy()
# #         except Exception as e:
# #             print(f"Error processed video paths {video_paths}: {e}")
# #             return []
#
# class VideoAudioFeatureExtractor:
#     """
#     Responsible for feature extraction from preprocessed video components and audio waveforms.
#     """
#     def extract_video_features(self, video_feature_extractor):
#         try:
#             # Execute Swin Transformer feature extraction
#             features = video_feature_extractor.execute_swin()
#             return features
#         except Exception as e:
#             print(e)
#             return []
#
#     def extract_audio_features(self, audio_preprocessor, video_paths, batch_size,save_path = None):
#         try:
#             # Process video paths as if they contain associated audio for waveform extraction
#             audio_features = audio_preprocessor.main_processing_waveforms(video_paths, batch_size,save_path = save_path)
#             return audio_features
#         except Exception as e:
#             print(f"Error extracting audio feature for {video_paths}: {e}")
#             return []
#
# class VideoAudioFeatureProcessor:
#     """
#     Combines component and feature extractors to produce a usable datasets.
#     """
#     def __init__(self, video_preprocess_dir,batch_size):
#         self.video_preprocess_dir = video_preprocess_dir
#         # self.feature_dir_vid = feature_dir_vid
#
#         # # Initialize the video preprocessor (FANET)
#         # self.video_preprocessor = VideoPreprocessor_PIPNet(
#         #     # batch_size=batch_size,
#         #     output_base_dir_real=video_save_dir,
#         #     real_output_txt_path=output_txt_file
#         # )
#
#         # Initialize audio preprocessor
#         self.audio_preprocessor = AudioPreprocessor()
#
#         # Initialize feature extractor (Swin Transformer)
#         self.video_feature_ext = VideoFeatureExtractor(video_preprocess_dir=video_preprocess_dir)
#
#         # self.component_extractor = VideoComponentExtractor()
#         self.feature_extractor = VideoAudioFeatureExtractor()
#         self.batch_size = batch_size
#
#     def create_datasubset(self, csv_path, use_preprocessed=True, video_paths=None, audio_paths = None, video_save_dir=None, output_txt_file=None):
#         processed_video_features = None
#         processed_audio_features = None
#         video_error = False
#         audio_error = False
#
#         # try:
#         #     # Extract components from raw videos
#         #     preprocessed_video_paths = self.component_extractor.extract_video_components(
#         #         video_paths, video_save_dir, output_txt_file, self.batch_size, self.video_preprocessor)
#         # except Exception as e:
#         #     print(f"Video Component Extraction Error: {e}")
#         #     video_error = True
#
#         try:
#             # Extract features from video if component extraction succeeded
#             # if not video_error:
#                 processed_video_features = self.feature_extractor.extract_video_features(self.video_feature_ext)
#         except Exception as e:
#             print(f"Video Feature Extraction Error: {e}")
#             video_error = True
#
#         try:
#             # Extract audio features using video file paths
#             processed_audio_features = self.feature_extractor.extract_audio_features(
#                 self.audio_preprocessor, video_paths, self.batch_size)
#         except Exception as e:
#             print(f"Audio Feature Extraction Error: {e}")
#             audio_error = True
#
#         # Return features only if both succeeded
#         if not audio_error:
#             return processed_audio_features, processed_video_features
#         else:
#             print("Errors encountered. No features returned.")
#             return None, None
#
# ###############################################################################
# # DATASET CLASS
# ###############################################################################
#
# class VideoAudioDataset(Dataset):
#     """
#     Custom PyTorch Dataset for loading video paths and labels.
#     """
#     def __init__(self, project_dir_curr, csv_name="training_data_two.csv", augmentations=None):
#         self.project_dir_curr = project_dir_curr
#         self.csv_name = csv_name
#         self.augmentations = augmentations
#
#         # Load paths and labels directly (audio paths removed)
#         self.video_paths,self.audio_paths, self.labels = create_file_paths(project_dir_curr, csv_name)
#
#         # Store data as list of tuples (video_path, label)
#         self.data = list(zip(self.video_paths,self.audio_paths, self.labels))
#
#     def __len__(self):
#         # Return total number of samples
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         # Get one sample from datasets
#         video_path, audio_path, label = self.data[idx]
#
#         # Apply optional augmentations
#         if self.augmentations:
#             video_path = self.augmentations(video_path)
#
#         # Return video path and label
#         return str(video_path),str(audio_path), label
#
# class VideoAudioDatasetEval(Dataset):
#     """
#     Custom PyTorch Dataset for loading video paths and labels for inference evaluation.
#     """
#     def __init__(self, project_dir_curr, csv_name="sampled_combined_data.csv", augmentations=None):
#         self.project_dir_curr = project_dir_curr
#         self.csv_name = csv_name
#         self.augmentations = augmentations
#
#         # Load paths and labels directly
#         self.video_paths, self.labels = create_file_paths_for_inference_eval(project_dir_curr, csv_name)
#
#         # Store data as list of tuples (video_path, label)
#         self.data = list(zip(self.video_paths, self.labels))
#
#     def __len__(self):
#         # Return total number of samples
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         # Get one sample from datasets
#         video_path, label = self.data[idx]
#
#         # Apply optional augmentations
#         if self.augmentations:
#             video_path = self.augmentations(video_path)
#
#         # Return video path and label
#         return str(video_path), label
#
# # if __name__ == '__main__':
# #     project_root = get_project_root()
# #     file_paths, file_paths_two, labels= create_file_paths(project_root,"training_data_two.csv")
# #     print(str(file_paths[1:20]))
# #     print(str(file_paths_two[1:20]))
# #
