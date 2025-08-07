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
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
import tempfile
import shutil

# NEW: Set optimal multiprocessing settings for CUDA
mp.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')


class VideoPreprocessor_FANET:
    # NEW: Class-level lock for face alignment synchronization
    _fa_lock = mp.Lock()

    def __init__(self, batch_size: int, output_base_dir: str = None,
                 device: str = 'cuda', rank: int = 0, stream=None):
        self.batch_size = batch_size
        self.output_base_dir = output_base_dir
        self.device = device
        self.rank = rank

        # NEW: Use process-specific stream for CUDA operations
        self.stream = stream or torch.cuda.current_stream()

        os.makedirs(self.output_base_dir, exist_ok=True)

        # NEW: Lazy initialization of face alignment to avoid CUDA context issues
        self.fa = None
        self._init_face_alignment()

        # OLD: Direct initialization caused CUDA context conflicts
        # self.fa = face_alignment.FaceAlignment(
        #     face_alignment.LandmarksType.TWO_D,
        #     device=self.device,
        #     face_detector='sfd',
        #     flip_input=False
        # )

        # NEW: Enable TF32 for A100 optimization
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # NEW: Increased thread workers for better parallelism on A100
        self.frame_executor = ThreadPoolExecutor(
            max_workers=4,  # OLD: was 2
            thread_name_prefix=f"gpu{rank}_frame_reader"
        )
        self.save_executor = ThreadPoolExecutor(
            max_workers=4,  # OLD: was 2
            thread_name_prefix=f"gpu{rank}_video_writer"
        )

        # NEW: Thread-local storage for video writers
        self.writer_locks = {}
        self.temp_dir = tempfile.mkdtemp(prefix=f"gpu{rank}_")

    def _init_face_alignment(self):
        """NEW: Lazy and thread-safe initialization of face alignment"""
        if self.fa is None:
            with self._fa_lock:
                if self.fa is None:  # Double-check pattern
                    torch.cuda.set_device(self.device)
                    self.fa = face_alignment.FaceAlignment(
                        face_alignment.LandmarksType.TWO_D,
                        device=self.device,
                        face_detector='sfd',
                        flip_input=False
                    )
                    print(f"[GPU {self.rank}] Initialized face alignment")

    def __call__(self, video_paths: list[str]) -> None:
        # NEW: Process videos in parallel batches for better GPU utilization
        batch_size = 2  # Process 2 videos simultaneously per GPU

        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i + batch_size]
            futures = []

            for path in batch_paths:
                future = self.frame_executor.submit(self.process_video, path)
                futures.append(future)

            # Wait for batch to complete before starting next
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ [GPU {self.rank}] Video processing failed: {e}")

        # OLD: Sequential processing
        # futures = []
        # for path in video_paths:
        #     future = self.frame_executor.submit(self.process_video, path)
        #     futures.append(future)
        # for future in futures:
        #     try:
        #         future.result()
        #     except Exception as e:
        #         print(f"❌ [GPU {self.rank}] Video processing failed: {e}")

        # Clean up
        self.save_executor.shutdown(wait=True)
        self.frame_executor.shutdown(wait=True)

        # NEW: Clean up temp directory
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def process_video(self, video_path: str) -> None:
        if not os.path.exists(video_path):
            print(f"❌ Video path does not exist: {video_path}")
            return

        video_name = Path(video_path).stem
        unique_id = str(uuid.uuid4())[:8]

        # NEW: Use .avi with MJPG for better compatibility
        out_path = os.path.join(self.output_base_dir, f"{video_name}_{unique_id}_lips.avi")

        # OLD: MP4 format caused issues
        # out_path = os.path.join(self.output_base_dir, f"{video_name}_{unique_id}_lips_only.mp4")

        print(f"[INFO] [GPU {self.rank}] Processing {video_path} -> {out_path}")

        try:
            # NEW: Larger queue sizes for A100 throughput
            frame_queue = Queue(maxsize=100)  # OLD: was 50
            crop_queue = Queue(maxsize=100)  # OLD: was 50

            # Start async frame reader
            reader_future = self.frame_executor.submit(
                self._async_frame_reader, video_path, frame_queue
            )

            # Start async video writer with new safe implementation
            writer_future = self.save_executor.submit(
                self._async_video_writer_safe, out_path, crop_queue
            )

            # Process frames in batches on GPU
            self._process_frame_stream_safe(frame_queue, crop_queue)

            # Signal end of processing
            crop_queue.put(None)

            # Wait for completion
            reader_future.result()
            writer_future.result()

            print(f"[GPU {self.rank}] ✅ Completed {video_path}")

        except Exception as e:
            print(f"❌ [GPU {self.rank}] Error processing {video_path}: {e}")
        finally:
            # NEW: Process-specific cleanup without global cache clear
            gc.collect()
            # OLD: This affected all processes!
            # torch.cuda.empty_cache()

    def _async_frame_reader(self, video_path: str, frame_queue: Queue):
        """Frame reader with better error handling"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            frame_queue.put(None)
            return

        try:
            batch = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    if batch:  # Send remaining frames
                        frame_queue.put(batch)
                    break

                batch.append(frame)
                frame_count += 1

                if len(batch) >= self.batch_size:
                    frame_queue.put(batch)
                    batch = []

            print(f"[GPU {self.rank}] Read {frame_count} frames from {video_path}")

        except Exception as e:
            print(f"❌ Frame reading error: {e}")
        finally:
            frame_queue.put(None)  # EOF signal
            cap.release()

    def _async_video_writer_safe(self, out_path: str, crop_queue: Queue):
        """NEW: Completely rewritten video writer for process safety"""
        frames_buffer = []
        frame_count = 0

        try:
            # Collect all frames first to avoid threading issues
            while True:
                crops = crop_queue.get()
                if crops is None:  # EOF signal
                    break

                for crop in crops:
                    if crop is not None:
                        # NEW: Make a copy to avoid memory view issues
                        frames_buffer.append(crop.copy())
                        frame_count += 1

                crop_queue.task_done()

            # Write all frames at once after collection
            if frames_buffer:
                h, w = frames_buffer[0].shape[:2]

                # NEW: Use temporary file for atomic write
                temp_path = os.path.join(self.temp_dir, f"temp_{uuid.uuid4()}.avi")

                # NEW: Use MJPG codec for better compatibility
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')

                # OLD: MP4V codec had issues
                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                out = cv2.VideoWriter(temp_path, fourcc, 25.0, (w, h))

                if not out.isOpened():
                    print(f"❌ [GPU {self.rank}] Failed to open video writer: {temp_path}")
                    return

                # Write all frames
                written = 0
                for frame in frames_buffer:
                    # NEW: Ensure consistent dimensions
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)

                    success = out.write(frame)
                    if success:
                        written += 1

                out.release()

                # NEW: Verify file before moving
                if written > 0 and os.path.getsize(temp_path) > 1000:  # At least 1KB
                    shutil.move(temp_path, out_path)
                    print(f"[GPU {self.rank}] ✅ Saved {written}/{frame_count} frames to {out_path}")
                else:
                    print(f"❌ [GPU {self.rank}] Invalid video file, not saving")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        except Exception as e:
            print(f"❌ [GPU {self.rank}] Video writing error: {e}")
            import traceback
            traceback.print_exc()

    # OLD: Previous video writer implementation
    # def _async_video_writer(self, out_path: str, crop_queue: Queue):
    #     """✅ Dedicated thread for writing video file"""
    #     out = None
    #     frame_count = 0
    #     try:
    #         while True:
    #             crops = crop_queue.get()
    #             if crops is None:  # EOF signal
    #                 break
    #             if out is None:
    #                 if not crops:
    #                     continue
    #                 h, w = crops[0].shape[:2]
    #                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #                 out = cv2.VideoWriter(out_path, fourcc, 25.0, (w, h))
    #                 if not out.isOpened():
    #                     print(f"❌ Failed to open video writer: {out_path}")
    #                     return
    #             for crop in crops:
    #                 if crop is not None:
    #                     out.write(crop)
    #                     frame_count += 1
    #             crop_queue.task_done()
    #     except Exception as e:
    #         print(f"❌ Video writing error for {out_path}: {e}")
    #     finally:
    #         if out is not None:
    #             out.release()
    #         print(f"[GPU {self.rank}] ✅ Saved {frame_count} frames to {out_path}")

    def _process_frame_stream_safe(self, frame_queue: Queue, crop_queue: Queue):
        """NEW: GPU processing with proper CUDA context isolation"""
        while True:
            batch = frame_queue.get()
            if batch is None:  # EOF signal
                break

            if not batch:  # Empty batch
                continue

            try:
                # NEW: Process with CUDA stream context
                with torch.cuda.stream(self.stream):
                    crops = self._process_batch_cuda_safe(batch)
                    if crops:
                        crop_queue.put(crops)
            except Exception as e:
                print(f"⚠️ [GPU {self.rank}] Batch processing error: {e}")
            finally:
                frame_queue.task_done()

    # OLD: Previous processing implementation
    # def _process_frame_stream(self, frame_queue: Queue, crop_queue: Queue):
    #     """✅ GPU processing pipeline with continuous streaming"""
    #     while True:
    #         batch = frame_queue.get()
    #         if batch is None:
    #             break
    #         if not batch:
    #             continue
    #         try:
    #             crops = self._process_batch_optimized(batch)
    #             if crops:
    #                 crop_queue.put(crops)
    #         except Exception as e:
    #             print(f"⚠️ [GPU {self.rank}] Batch processing error: {e}")
    #         finally:
    #             frame_queue.task_done()

    def _process_batch_cuda_safe(self, frame_batch):
        """NEW: Completely rewritten batch processing with CUDA safety"""
        if not frame_batch:
            return []

        crops = []

        try:
            # NEW: Dynamic batch size based on available GPU memory
            mem_free = torch.cuda.mem_get_info(self.device)[0] / 1024 ** 3  # GB
            sub_batch_size = min(4 if mem_free < 40 else 8, len(frame_batch))

            # OLD: Fixed batch size
            # sub_batch_size = min(8, len(frame_batch))

            for i in range(0, len(frame_batch), sub_batch_size):
                sub_batch = frame_batch[i:i + sub_batch_size]

                # Convert frames to tensors
                frame_tensors = []
                valid_indices = []

                for idx, frame in enumerate(sub_batch):
                    if frame is not None:
                        # NEW: Ensure frame is contiguous in memory
                        frame_copy = np.ascontiguousarray(frame)
                        frame_tensor = torch.from_numpy(frame_copy).permute(2, 0, 1).float()
                        frame_tensors.append(frame_tensor)
                        valid_indices.append(idx)

                if not frame_tensors:
                    continue

                # Stack and move to GPU with stream
                frame_batch_tensor = torch.stack(frame_tensors, dim=0)

                # NEW: Non-blocking transfer with explicit stream
                frame_batch_tensor = frame_batch_tensor.to(
                    self.device, non_blocking=True
                )

                # NEW: Synchronize before face detection
                torch.cuda.synchronize(self.device)

                # Get landmarks with process lock
                with self._fa_lock:
                    landmarks_batch = self.fa.get_landmarks_from_batch(frame_batch_tensor)

                # Process each frame
                for j, (frame_tensor, landmarks) in enumerate(zip(frame_batch_tensor, landmarks_batch)):
                    try:
                        if landmarks is None:
                            continue

                        # NEW: Immediately move to CPU to free GPU memory
                        frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                        # Extract lip segment
                        lip_crop, _ = self.extract_lip_segment(frame_np, landmarks)

                        if lip_crop is not None:
                            # Resize to standard size
                            resized_crop = cv2.resize(
                                lip_crop, (224, 224),
                                interpolation=cv2.INTER_CUBIC
                            )
                            crops.append(resized_crop)

                    except Exception as e:
                        print(f"⚠️ Frame processing error: {e}")
                        continue

                # NEW: Explicit tensor cleanup without global cache clear
                del frame_batch_tensor
                del frame_tensors
                del landmarks_batch

                # NEW: Small delay to prevent GPU overload
                if i + sub_batch_size < len(frame_batch):
                    time.sleep(0.001)

        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # NEW: Local garbage collection only
            gc.collect()
            # OLD: This was problematic!
            # torch.cuda.empty_cache()

        return crops

    # OLD: Previous batch processing implementation
    # def _process_batch_optimized(self, frame_batch):
    #     """✅ Optimized batch processing with better memory management"""
    #     if not frame_batch:
    #         return []
    #     crops = []
    #     frame_batch_tensor = None
    #     try:
    #         sub_batch_size = min(8, len(frame_batch))
    #         for i in range(0, len(frame_batch), sub_batch_size):
    #             sub_batch = frame_batch[i:i + sub_batch_size]
    #             frame_tensors = []
    #             for frame in sub_batch:
    #                 if frame is not None:
    #                     frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
    #                     frame_tensors.append(frame_tensor)
    #             if not frame_tensors:
    #                 continue
    #             frame_batch_tensor = torch.stack(frame_tensors, dim=0).to(self.device)
    #             landmarks_batch = self.fa.get_landmarks_from_batch(frame_batch_tensor)
    #             for j, (frame_tensor, landmarks) in enumerate(zip(frame_batch_tensor, landmarks_batch)):
    #                 try:
    #                     if landmarks is None:
    #                         continue
    #                     frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    #                     lip_crop, _ = self.extract_lip_segment(frame_np, landmarks)
    #                     if lip_crop is not None:
    #                         resized_crop = cv2.resize(lip_crop, (224, 224), interpolation=cv2.INTER_CUBIC)
    #                         crops.append(resized_crop)
    #                 except Exception as e:
    #                     print(f"⚠️ Frame processing error: {e}")
    #                     continue
    #             del frame_batch_tensor, frame_tensors, landmarks_batch
    #             torch.cuda.empty_cache()
    #     except Exception as e:
    #         print(f"❌ Batch processing failed: {e}")
    #     finally:
    #         if frame_batch_tensor is not None:
    #             del frame_batch_tensor
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #     return crops

    def extract_lip_segment(self, frame, landmarks):
        """Same implementation - no changes needed"""
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


def worker_process(rank, chunks, batch_size, output_dir, return_dict):
    """NEW: Enhanced worker process with proper CUDA isolation"""
    try:
        # NEW: Set CUDA device before ANY CUDA operations
        torch.cuda.set_device(rank)
        torch.cuda.init()  # Force CUDA initialization

        device_str = f'cuda:{rank}'

        # NEW: Create process-specific CUDA stream
        stream = torch.cuda.Stream(device=rank)

        # NEW: Set process priority for better scheduling
        os.nice(10)  # Lower priority to prevent GPU hogging

        with torch.cuda.device(rank):
            with torch.cuda.stream(stream):
                processor = VideoPreprocessor_FANET(
                    batch_size=batch_size,
                    output_base_dir=output_dir,
                    device=device_str,
                    rank=rank,
                    stream=stream  # NEW: Pass stream to processor
                )

                assigned_videos = chunks[rank]
                num_videos = len(assigned_videos)

                print(f"[GPU {rank}] Starting {num_videos} videos.")
                start_time = time.time()

                processor(assigned_videos)

                elapsed = time.time() - start_time
                print(f"[GPU {rank}] ✅ Completed all {num_videos} videos in {elapsed / 60:.2f} minutes")

                return_dict[rank] = True

    except Exception as e:
        print(f"❌ [GPU {rank}] Worker failed: {e}")
        import traceback
        traceback.print_exc()
        return_dict[rank] = False
    finally:
        # NEW: Proper cleanup
        gc.collect()
        # Don't use global cache clear
        # torch.cuda.empty_cache()


# OLD: Previous worker implementation
# def worker_process(rank, chunks, batch_size, output_dir, return_dict):
#     """✅ Same worker process with better error handling"""
#     torch.cuda.set_device(rank)
#     device_str = f'cuda:{rank}'
#     processor = VideoPreprocessor_FANET(
#         batch_size=batch_size,
#         output_base_dir=output_dir,
#         device=device_str,
#         rank=rank
#     )
#     assigned_videos = chunks[rank]
#     num_videos = len(assigned_videos)
#     print(f"[GPU {rank}] Starting {num_videos} videos.")
#     start_time = time.time()
#     try:
#         processor(assigned_videos)
#         return_dict[rank] = True
#         elapsed = time.time() - start_time
#         print(f"[GPU {rank}] ✅ Completed all {num_videos} videos in {elapsed/60:.2f} minutes")
#     except Exception as e:
#         print(f"❌ [GPU {rank}] Worker failed: {e}")
#         return_dict[rank] = False
#     finally:
#         del processor
#         gc.collect()
#         torch.cuda.empty_cache()


def get_video_info(video_path):
    """NEW: Helper to get video duration for better load balancing"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration = frame_count / fps if fps > 0 else 0
        return duration
    except:
        return 0


def balanced_chunk_videos(video_paths, num_gpus):
    """NEW: Intelligent load balancing based on video duration"""
    # Get duration for each video
    videos_with_info = []
    for path in video_paths:
        duration = get_video_info(path)
        videos_with_info.append((path, duration))

    # Sort by duration (longest first)
    videos_with_info.sort(key=lambda x: x[1], reverse=True)

    # Initialize chunks with total duration tracking
    chunks = [[] for _ in range(num_gpus)]
    chunk_durations = [0] * num_gpus

    # Distribute videos to balance total duration
    for video_path, duration in videos_with_info:
        # Find GPU with least total duration
        min_idx = chunk_durations.index(min(chunk_durations))
        chunks[min_idx].append(video_path)
        chunk_durations[min_idx] += duration

    return chunks


def parallel_main(video_paths: list[str], batch_size: int, output_dir: str):
    """NEW: Enhanced main function with better GPU management"""
    available_gpus = torch.cuda.device_count()
    world_size = min(available_gpus, 8)

    # NEW: Set environment for optimal multi-GPU performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    manager = mp.Manager()
    return_dict = manager.dict()

    # NEW: Use intelligent load balancing
    chunks = balanced_chunk_videos(video_paths, world_size)

    # OLD: Simple round-robin distribution
    # chunks = [video_paths[i::world_size] for i in range(world_size)]

    print(f"🚀 Starting distributed processing on {world_size} GPUs")
    print(f"📁 Videos per GPU: {[len(chunk) for chunk in chunks]}")

    # NEW: Print estimated durations per GPU
    for i, chunk in enumerate(chunks):
        total_duration = sum(get_video_info(p) for p in chunk)
        print(f"  GPU {i}: {len(chunk)} videos, ~{total_duration / 60:.1f} minutes total")

    mp.spawn(
        worker_process,
        args=(chunks, batch_size, output_dir, return_dict),
        nprocs=world_size,
        join=True
    )

    successful_gpus = sum(return_dict.values())
    print(f"✅ Processing complete. {successful_gpus}/{world_size} GPUs finished successfully.")

    # NEW: Verify output files
    total_files = len([f for f in os.listdir(output_dir) if f.endswith('.avi')])
    print(f"📊 Total output files created: {total_files}/{len(video_paths)}")


# # Example usage:
# if __name__ == "__main__":
#     # Your video paths
#     video_paths = [...]  # List of video file paths
#
#     # NEW: Optimal batch size for A100
#     batch_size = 16  # Increased from default
#
#     output_dir = "output_lips"
#
#     parallel_main(video_paths, batch_size, output_dir)