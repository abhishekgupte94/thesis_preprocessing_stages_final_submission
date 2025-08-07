import os
import gc
import cv2
import torch
import face_alignment
import numpy as np
import torch.multiprocessing as mp
from pathlib import Path
import time
from queue import Queue
from threading import Thread


class VideoPreprocessor_FANET:
    """
    Distributed video lip-extraction using FaceAlignment.
    Uses async .mp4 saving to avoid blocking GPU.
    """

    def __init__(self, batch_size: int, output_base_dir: str = None, device: str = 'cuda', rank: int = 0):
        self.batch_size = batch_size
        self.output_base_dir = output_base_dir
        self.device = device
        self.rank = rank

        os.makedirs(self.output_base_dir, exist_ok=True)

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=self.device,
            face_detector='sfd',
            flip_input=False
        )

        torch.backends.cudnn.benchmark = True

        # ✅ Async saving setup
        self.save_queue = Queue(maxsize=10)
        self.saver_thread = Thread(target=self._save_worker, daemon=True)
        self.saver_thread.start()

    def __call__(self, video_paths: list[str]) -> None:
        for path in video_paths:
            self.process_video(path)
        self.save_queue.join()  # ✅ Wait until all saves are complete

    def process_video(self, video_path: str) -> None:
        if not os.path.exists(video_path):
            print(f"❌ Video path does not exist: {video_path}")
            return

        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return

        out_path = os.path.join(self.output_base_dir, f"{video_name}_lips_only.mp4")  # ✅ Save as .mp4
        print(f"[INFO] [GPU {self.rank}] Saving intermediate output to {out_path}")

        buffer = []
        all_crops = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            buffer.append(frame)
            if len(buffer) >= self.batch_size:
                crops = self._process_batch(buffer)
                all_crops.extend(crops)
                buffer.clear()

        if buffer:
            crops = self._process_batch(buffer)
            all_crops.extend(crops)

        cap.release()

        if all_crops:
            self.save_queue.put((out_path, all_crops))  # ✅ Async saving
            print(f"[GPU {self.rank}] Queued {len(all_crops)} crops to {out_path}")
        else:
            print(f"[GPU {self.rank}] No valid crops extracted from {video_path}")

        del cap, buffer, all_crops
        gc.collect()
        torch.cuda.empty_cache()

    def _process_batch(self, frame_batch):
        crops = []
        try:
            frame_batch_tensor = torch.stack(
                [torch.from_numpy(frame).permute(2, 0, 1) for frame in frame_batch],
                dim=0
            ).float().to(self.device)

            landmarks_batch = self.fa.get_landmarks_from_batch(frame_batch_tensor)

            for frame_tensor, landmarks in zip(frame_batch_tensor, landmarks_batch):
                try:
                    if landmarks is None:
                        print(f"⚠️ Skipping frame: no face detected.")
                        continue

                    frame = frame_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    lip_crop, _ = self.extract_lip_segment(frame, landmarks)

                    if lip_crop is None:
                        print(f"⚠️ Skipping frame: invalid lip crop.")
                        continue

                    resized_crop = cv2.resize(lip_crop, (224, 224), interpolation=cv2.INTER_CUBIC)

                    if not isinstance(resized_crop, np.ndarray) or resized_crop.ndim != 3 or resized_crop.dtype != np.uint8:
                        print(f"⚠️ Invalid frame for saving. Skipping.")
                        continue

                    crops.append(resized_crop)

                except Exception as e:
                    print(f"⚠️ Error during frame processing: {e}")

        except Exception as e:
            print(f"❌ Batch processing failed: {e}")

        finally:
            del frame_batch, frame_batch_tensor, landmarks_batch
            gc.collect()
            torch.cuda.empty_cache()

        return crops

    def extract_lip_segment(self, frame, landmarks):
        if landmarks is None:
            return None, (0, 0, 0, 0)

        lip_landmarks = landmarks[48:]
        x_coords = lip_landmarks[:, 0].astype(int)
        y_coords = lip_landmarks[:, 1].astype(int)

        x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
        y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)

        if x_max <= x_min or y_max <= y_min or (x_max - x_min) < 10 or (y_max - y_min) < 10:
            return None, (x_min, y_min, x_max, y_max)

        lip_crop = frame[y_min:y_max, x_min:x_max]
        return lip_crop, (x_min, y_min, x_max, y_max)

    def _save_worker(self):
        """ ✅ Background thread that saves .mp4 datasets asynchronously """
        while True:
            out_path, frames = self.save_queue.get()
            try:
                tmp_path = out_path + '.tmp'  # temp file to avoid corruption
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(tmp_path, fourcc, 25, (224, 224))

                for frame in frames:
                    out.write(frame)

                out.release()
                os.rename(tmp_path, out_path)  # finalize file
                print(f"[GPU {self.rank}] ✅ Saved MP4: {out_path}")

            except Exception as e:
                print(f"❌ [GPU {self.rank}] Error saving MP4 to {out_path}: {e}")

            finally:
                self.save_queue.task_done()


def worker_process(rank, chunks, batch_size, output_dir, return_dict):
    torch.cuda.set_device(rank)
    device_str = f'cuda:{rank}'

    processor = VideoPreprocessor_FANET(
        batch_size=batch_size,
        output_base_dir=output_dir,
        device=device_str,
        rank=rank
    )

    assigned_videos = chunks[rank]
    num_videos = len(assigned_videos)

    print(f"[GPU {rank}] Starting {num_videos} videos.")

    start_time = time.time()

    for idx, video_path in enumerate(assigned_videos):
        try:
            processor.process_video(video_path)
        except Exception as e:
            print(f"⚠️ [GPU {rank}] Error processing {video_path}: {e}")

        if (idx + 1) % 10 == 0 or (idx + 1) == num_videos:
            elapsed = time.time() - start_time
            avg_time_per_video = elapsed / (idx + 1)
            eta_seconds = avg_time_per_video * (num_videos - idx - 1)
            print(f"[GPU {rank}] {idx + 1}/{num_videos} videos done. ETA: {eta_seconds / 60:.2f} minutes.")

    processor.save_queue.join()  # ✅ Ensure all video saves are complete
    return_dict[rank] = True

    del processor
    gc.collect()
    torch.cuda.empty_cache()


def parallel_main(video_paths: list[str], batch_size: int, output_dir: str):
    available_gpus = torch.cuda.device_count()
    world_size = available_gpus
    manager = mp.Manager()
    return_dict = manager.dict()
    chunks = [video_paths[i::world_size] for i in range(world_size)]

    mp.spawn(
        worker_process,
        args=(chunks, batch_size, output_dir, return_dict),
        nprocs=world_size,
        join=True
    )

    print(f"✅ All videos processed and saved to '{output_dir}'.")


# # ✅ Example usage block (add this to your main script)
# if __name__ == '__main__':
#     import glob
#
#     # 🗂️ Collect all video paths (modify the path as needed)
#     video_folder = '/path/to/your/videos'
#     video_paths = glob.glob(os.path.join(video_folder, '*.mp4'))  # or .avi, .mov etc.
#
#     # 📁 Output directory for processed videos
#     output_dir = '/path/to/save/processed_lips'
#
#     # 🧪 Batch size for processing frames
#     batch_size = 8
#
#     # 🚀 Start the distributed video processing pipeline
#     parallel_main(video_paths, batch_size, output_dir)
#
#     print("🎉 Lip-only video processed complete!")
