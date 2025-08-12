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
import json
import psutil

mp.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')


def load_storage_config():
    """
    Load storage configuration from pre-flight checks
    """
    if os.path.exists('storage_config.json'):
        with open('storage_config.json', 'r') as f:
            return json.load(f)
    else:
        # Fallback: auto-detect if no config exists
        return {
            'nvme_found': os.path.exists('/scratch') or os.path.exists('/tmp'),
            'writable_paths': ['/tmp'],
            'performance': {'/tmp': {'read_speed': 500, 'write_speed': 500}}
        }


class VideoPreprocessor_FANET:
    def __init__(self, batch_size: int, output_base_dir: str = None,
                 device: str = 'cuda', rank: int = 0, stream=None,
                 use_nvme_cache: bool = True):

        self.batch_size = batch_size
        self.output_base_dir = output_base_dir
        self.device = device
        self.rank = rank
        self.stream = stream or torch.cuda.current_stream()

        # NVMe cache management attributes
        self.use_nvme_cache = use_nvme_cache
        self.cache_dir = None
        self.cached_videos = {}

        # Initialize NVMe cache if enabled
        if self.use_nvme_cache:
            self._setup_nvme_cache()

        os.makedirs(self.output_base_dir, exist_ok=True)

        # Initialize thread-local storage for face alignment
        self._thread_local = threading.local()
        self.fa = None  # Remove the global face alignment instance
        self._init_face_alignment_main_thread()  # Initialize for main thread

        # CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Increased thread workers for better parallelism
        self.frame_executor = ThreadPoolExecutor(
            max_workers=16 if use_nvme_cache else 8,
            thread_name_prefix=f"gpu{rank}_frame_reader"
        )
        self.save_executor = ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix=f"gpu{rank}_video_writer"
        )

        self.writer_locks = {}
        self.temp_dir = tempfile.mkdtemp(prefix=f"gpu{rank}_")

    def _setup_nvme_cache(self):
        """
        Automatically detect and setup NVMe cache with proper locking
        """
        config = load_storage_config()

        # Priority order for Lambda Labs infrastructure
        cache_priorities = [
            '/scratch',  # Primary NVMe mount on Lambda
            '/tmp',  # Often NVMe or tmpfs (RAM disk)
            '/local',  # Sometimes local SSD
            '/dev/shm'  # Shared memory (RAM disk fallback)
        ]

        # Find fastest available storage
        best_cache = None
        best_speed = 0

        for path in cache_priorities:
            if path in config['performance']:
                speed = config['performance'][path]['read_speed']
                if speed > best_speed and os.path.exists(path):
                    # Check available space using psutil
                    usage = psutil.disk_usage(path)
                    free_gb = usage.free / (1024 ** 3)

                    if free_gb > 10:  # Need at least 10GB free
                        best_cache = path
                        best_speed = speed

        if best_cache:
            # Create cache directory with proper locking to avoid concurrent access issues
            cache_lock_file = Path(best_cache) / ".cache_setup.lock"
            cache_lock_file.parent.mkdir(exist_ok=True, parents=True)

            with open(cache_lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    # Create GPU-specific cache directory to avoid conflicts
                    self.cache_dir = Path(best_cache) / f"lip_extraction_cache_gpu{self.rank}"
                    self.cache_dir.mkdir(exist_ok=True, parents=True)

                    print(f"[GPU {self.rank}] ✅ NVMe cache enabled at {self.cache_dir}")
                    print(f"[GPU {self.rank}]    Speed: {best_speed:.0f} MB/s")
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        else:
            print(f"[GPU {self.rank}] ⚠️ No suitable NVMe storage found, using direct I/O")
            self.use_nvme_cache = False

    def _cache_videos_batch(self, video_paths: list) -> dict:
        """
        Pre-cache all videos to NVMe storage for fast access
        """
        if not self.use_nvme_cache:
            return {p: p for p in video_paths}

        print(f"[GPU {self.rank}] Caching {len(video_paths)} videos to NVMe...")
        start_time = time.time()

        cached_paths = {}

        def copy_to_cache(video_path):
            """Helper function for parallel copying"""
            try:
                video_name = Path(video_path).name
                cached_path = self.cache_dir / video_name

                # Skip if already cached (useful for retries)
                if not cached_path.exists():
                    shutil.copy2(video_path, cached_path)

                return video_path, str(cached_path)
            except Exception as e:
                print(f"⚠️ Failed to cache {video_path}: {e}")
                return video_path, video_path  # Fallback to original

        # Parallel copy with 16 workers for maximum I/O throughput
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(copy_to_cache, path) for path in video_paths]

            for i, future in enumerate(futures):
                orig_path, cached_path = future.result()
                cached_paths[orig_path] = cached_path

                # Progress reporting every 100 videos
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"[GPU {self.rank}] Cached {i + 1}/{len(video_paths)} "
                          f"({rate:.1f} videos/sec)")

        # Calculate and report caching statistics
        cache_time = time.time() - start_time
        cache_size = sum(os.path.getsize(p) for p in cached_paths.values()) / (1024 ** 3)

        print(f"[GPU {self.rank}] ✅ Cached {len(cached_paths)} videos in {cache_time:.1f}s")
        print(f"[GPU {self.rank}]    Cache size: {cache_size:.2f} GB")
        print(f"[GPU {self.rank}]    Cache speed: {cache_size * 1024 / cache_time:.0f} MB/s")

        return cached_paths

    def _init_face_alignment_main_thread(self):
        """Initialize face alignment for main thread only"""
        torch.cuda.set_device(self.device)
        # This will be accessed through _get_face_alignment() for thread safety
        print(f"[GPU {self.rank}] Initialized face alignment system")

    def _get_face_alignment(self):
        """Get thread-specific face alignment instance for CUDA thread safety"""
        if not hasattr(self._thread_local, 'fa'):
            # Ensure we're in the right CUDA context
            with torch.cuda.device(self.device):
                torch.cuda.set_device(self.device)
                self._thread_local.fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D,
                    device=self.device,
                    face_detector='sfd',
                    flip_input=False
                )
                print(
                    f"[GPU {self.rank}] Created thread-local face alignment for thread {threading.current_thread().name}")
        return self._thread_local.fa

    def __call__(self, video_paths: list[str]) -> None:
        # Pre-cache all assigned videos to NVMe before processing
        if self.use_nvme_cache:
            self.cached_videos = self._cache_videos_batch(video_paths)
            processing_paths = list(self.cached_videos.values())
        else:
            processing_paths = video_paths

        # Increased batch size for better GPU utilization
        batch_size = 8

        for i in range(0, len(processing_paths), batch_size):
            batch_paths = processing_paths[i:i + batch_size]
            futures = []

            for path in batch_paths:
                # Track original path for output naming
                orig_path = video_paths[i + batch_paths.index(path)]
                future = self.frame_executor.submit(
                    self.process_video, path, orig_path
                )
                futures.append(future)

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ [GPU {self.rank}] Video processing failed: {e}")

        # Ensure all processing is complete before cleanup
        self.save_executor.shutdown(wait=True)
        self.frame_executor.shutdown(wait=True)

        # Clean up NVMe cache after all processing is done
        if self.use_nvme_cache and self.cache_dir:
            try:
                print(f"[GPU {self.rank}] Cleaning up cache directory...")
                shutil.rmtree(self.cache_dir)
            except Exception as e:
                print(f"⚠️ Failed to clean cache: {e}")

        # Clean up temp directory
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def process_video(self, video_path: str, original_path: str = None) -> None:
        """
        Process a single video with improved error handling
        """
        if not os.path.exists(video_path):
            print(f"❌ Video path does not exist: {video_path}")
            return

        # Use original path for output naming if provided
        if original_path:
            video_name = Path(original_path).stem
        else:
            video_name = Path(video_path).stem

        unique_id = str(uuid.uuid4())[:8]
        out_path = os.path.join(self.output_base_dir, f"{video_name}_{unique_id}_lips.avi")

        # Log whether using cache or direct I/O
        if self.use_nvme_cache and video_path != original_path:
            print(f"[INFO] [GPU {self.rank}] Processing from cache: {video_path}")
        else:
            print(f"[INFO] [GPU {self.rank}] Processing: {video_path}")

        try:
            # Increased queue sizes for better buffering
            frame_queue = Queue(maxsize=500)
            crop_queue = Queue(maxsize=500)

            reader_future = self.frame_executor.submit(
                self._async_frame_reader, video_path, frame_queue
            )

            writer_future = self.save_executor.submit(
                self._async_video_writer_safe, out_path, crop_queue
            )

            self._process_frame_stream_safe(frame_queue, crop_queue)

            crop_queue.put(None)

            reader_future.result()
            writer_future.result()

            print(f"[GPU {self.rank}] ✅ Completed {video_name}")

        except Exception as e:
            print(f"❌ [GPU {self.rank}] Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()

    def _async_frame_reader(self, video_path: str, frame_queue: Queue):
        """Optimized frame reader for NVMe speeds"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            frame_queue.put(None)
            return

        # Increase buffer size for NVMe to reduce syscalls
        if self.use_nvme_cache:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

        try:
            batch = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    if batch:
                        frame_queue.put(batch)
                    break

                batch.append(frame)
                frame_count += 1

                if len(batch) >= self.batch_size:
                    frame_queue.put(batch)
                    batch = []

            print(f"[GPU {self.rank}] Read {frame_count} frames")

        except Exception as e:
            print(f"❌ Frame reading error: {e}")
        finally:
            frame_queue.put(None)
            cap.release()

    def _async_video_writer_safe(self, out_path: str, crop_queue: Queue):
        """Improved video writer with better codec and error handling"""
        frames_buffer = []
        frame_count = 0

        try:
            while True:
                crops = crop_queue.get()
                if crops is None:
                    break

                for crop in crops:
                    if crop is not None:
                        frames_buffer.append(crop.copy())
                        frame_count += 1

                crop_queue.task_done()

            if frames_buffer:
                h, w = frames_buffer[0].shape[:2]

                # Write to NVMe cache first for faster writes
                if self.use_nvme_cache and self.cache_dir:
                    temp_path = str(self.cache_dir / f"temp_{uuid.uuid4()}.avi")
                else:
                    temp_path = os.path.join(self.temp_dir, f"temp_{uuid.uuid4()}.avi")

                # Use better codec for compatibility and performance
                # Try multiple codecs in order of preference
                codecs = [
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    cv2.VideoWriter_fourcc(*'XVID'),
                    cv2.VideoWriter_fourcc(*'MJPG')  # Fallback
                ]

                out = None
                for fourcc in codecs:
                    out = cv2.VideoWriter(temp_path, fourcc, 25.0, (w, h))
                    if out.isOpened():
                        break
                    out.release()

                if not out or not out.isOpened():
                    print(f"❌ [GPU {self.rank}] Failed to open video writer with any codec")
                    return

                written = 0
                for frame in frames_buffer:
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)

                    success = out.write(frame)
                    if success:
                        written += 1

                out.release()

                # Enhanced error diagnostics
                if written > 0:
                    file_size = os.path.getsize(temp_path)
                    print(f"[GPU {self.rank}] Debug: Written {written} frames, file size: {file_size} bytes")

                    if file_size > 1000:  # Minimum reasonable file size
                        shutil.move(temp_path, out_path)
                        print(f"[GPU {self.rank}] ✅ Saved {written}/{frame_count} frames")
                    else:
                        print(f"❌ [GPU {self.rank}] Invalid video file - size too small: {file_size} bytes")
                        print(f"    Frame dimensions: {w}x{h}")
                        print(f"    Frames in buffer: {len(frames_buffer)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                else:
                    print(f"❌ [GPU {self.rank}] No frames written successfully")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        except Exception as e:
            print(f"❌ [GPU {self.rank}] Video writing error: {e}")
            import traceback
            traceback.print_exc()

    def _process_frame_stream_safe(self, frame_queue: Queue, crop_queue: Queue):
        """Process frame batches with improved parallelism"""
        while True:
            batch = frame_queue.get()
            if batch is None:
                break

            if not batch:
                continue

            try:
                with torch.cuda.stream(self.stream):
                    crops = self._process_batch_cuda_safe(batch)
                    if crops:
                        crop_queue.put(crops)
            except Exception as e:
                print(f"⚠️ [GPU {self.rank}] Batch processing error: {e}")
            finally:
                frame_queue.task_done()

    def _process_batch_cuda_safe(self, frame_batch):
        """Process batch with no lock contention"""
        if not frame_batch:
            return []

        crops = []

        try:
            mem_free = torch.cuda.mem_get_info(self.device)[0] / 1024 ** 3
            sub_batch_size = min(16 if mem_free < 40 else 32, len(frame_batch))

            for i in range(0, len(frame_batch), sub_batch_size):
                sub_batch = frame_batch[i:i + sub_batch_size]

                frame_tensors = []
                valid_indices = []

                for idx, frame in enumerate(sub_batch):
                    if frame is not None:
                        frame_copy = np.ascontiguousarray(frame)
                        frame_tensor = torch.from_numpy(frame_copy).permute(2, 0, 1).float()
                        frame_tensors.append(frame_tensor)
                        valid_indices.append(idx)

                if not frame_tensors:
                    continue

                frame_batch_tensor = torch.stack(frame_tensors, dim=0)
                frame_batch_tensor = frame_batch_tensor.to(
                    self.device, non_blocking=True
                )

                torch.cuda.synchronize(self.device)

                # Get thread-safe face alignment instance
                fa = self._get_face_alignment()
                landmarks_batch = fa.get_landmarks_from_batch(frame_batch_tensor)

                for j, (frame_tensor, landmarks) in enumerate(zip(frame_batch_tensor, landmarks_batch)):
                    try:
                        if landmarks is None:
                            continue

                        frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        lip_crop, _ = self.extract_lip_segment(frame_np, landmarks)

                        if lip_crop is not None:
                            resized_crop = cv2.resize(
                                lip_crop, (224, 224),
                                interpolation=cv2.INTER_CUBIC
                            )
                            crops.append(resized_crop)

                    except Exception as e:
                        print(f"⚠️ Frame processing error: {e}")
                        continue

                del frame_batch_tensor
                del frame_tensors
                del landmarks_batch

                if i + sub_batch_size < len(frame_batch):
                    time.sleep(0.001)

        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()

        return crops

    def extract_lip_segment(self, frame, landmarks):
        """Extract lip region from frame using landmarks"""
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


def worker_process(rank, chunks, batch_size, output_dir, return_dict, use_nvme):
    """Worker process with improved error handling"""
    try:
        torch.cuda.set_device(rank)
        torch.cuda.init()

        device_str = f'cuda:{rank}'
        stream = torch.cuda.Stream(device=rank)

        # Lower process priority for better system responsiveness
        os.nice(10)

        with torch.cuda.device(rank):
            with torch.cuda.stream(stream):
                processor = VideoPreprocessor_FANET(
                    batch_size=batch_size,
                    output_base_dir=output_dir,
                    device=device_str,
                    rank=rank,
                    stream=stream,
                    use_nvme_cache=use_nvme
                )

                assigned_videos = chunks[rank]
                num_videos = len(assigned_videos)

                print(f"[GPU {rank}] Starting {num_videos} videos.")
                start_time = time.time()

                processor(assigned_videos)

                elapsed = time.time() - start_time
                throughput = num_videos / elapsed

                print(f"[GPU {rank}] ✅ Completed {num_videos} videos in {elapsed / 60:.2f} min")
                print(f"[GPU {rank}]    Throughput: {throughput:.2f} videos/sec")

                return_dict[rank] = True

    except Exception as e:
        print(f"❌ [GPU {rank}] Worker failed: {e}")
        import traceback
        traceback.print_exc()
        return_dict[rank] = False
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def parallel_main(video_paths: list[str], batch_size: int, output_dir: str,
                  use_nvme: bool = True):
    """Main entry point with NVMe support"""

    # Check for NVMe availability before processing
    if use_nvme:
        print("🔍 Checking NVMe availability...")
        config = load_storage_config()

        if config['nvme_found']:
            print("✅ NVMe storage detected and will be used for caching")
        else:
            print("⚠️ No NVMe storage found, falling back to direct I/O")
            use_nvme = False

    available_gpus = torch.cuda.device_count()
    world_size = min(available_gpus, 8)

    # Environment optimizations
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    manager = mp.Manager()
    return_dict = manager.dict()

    chunks = balanced_chunk_videos(video_paths, world_size)

    print(f"🚀 Starting distributed processing on {world_size} GPUs")
    print(f"📁 Videos per GPU: {[len(chunk) for chunk in chunks]}")

    if use_nvme:
        print(f"💾 NVMe caching enabled")

    for i, chunk in enumerate(chunks):
        total_duration = sum(get_video_info(p) for p in chunk)
        print(f"  GPU {i}: {len(chunk)} videos, ~{total_duration / 60:.1f} minutes total")

    mp.spawn(
        worker_process,
        args=(chunks, batch_size, output_dir, return_dict, use_nvme),
        nprocs=world_size,
        join=True
    )

    successful_gpus = sum(return_dict.values())
    print(f"✅ Processing complete. {successful_gpus}/{world_size} GPUs finished successfully.")

    total_files = len([f for f in os.listdir(output_dir) if f.endswith('.avi')])
    print(f"📊 Total output files created: {total_files}/{len(video_paths)}")


def get_video_info(video_path):
    """Get video duration for load balancing"""
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
    """Balance video workload across GPUs by duration"""
    videos_with_info = []
    for path in video_paths:
        duration = get_video_info(path)
        videos_with_info.append((path, duration))

    # Sort by duration for better load balancing
    videos_with_info.sort(key=lambda x: x[1], reverse=True)

    chunks = [[] for _ in range(num_gpus)]
    chunk_durations = [0] * num_gpus

    # Assign videos to GPU with least total duration
    for video_path, duration in videos_with_info:
        min_idx = chunk_durations.index(min(chunk_durations))
        chunks[min_idx].append(video_path)
        chunk_durations[min_idx] += duration

    return chunks


# Usage example:
# if __name__ == "__main__":
#     # Example usage
#     video_list = ["path/to/video1.mp4", "path/to/video2.mp4"]  # Add your video paths
#     output_directory = "./output_lips"
#
#     # Run with default settings (NVMe enabled if available)
#     parallel_main(
#         video_paths=video_list,
#         batch_size=32,  # Increased from 16 for better GPU utilization
#         output_dir=output_directory,
#         use_nvme=True
#     )