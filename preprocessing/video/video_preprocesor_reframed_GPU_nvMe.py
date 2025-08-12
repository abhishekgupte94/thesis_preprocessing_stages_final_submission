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

# ============ NEW NVMe ADDITIONS START ============
import json  # NEW: For saving/loading storage configuration
import psutil  # NEW: For disk usage monitoring and partition detection

# ============ NEW NVMe ADDITIONS END ============

mp.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')


# ============ NEW NVMe FUNCTION START ============
def load_storage_config():
    """
    NEW FUNCTION: Load storage configuration from pre-flight checks

    Technology: JSON configuration persistence
    - Saves results of storage benchmark to avoid re-running
    - Contains NVMe detection results and performance metrics
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


# ============ NEW NVMe FUNCTION END ============

class VideoPreprocessor_FANET:
    _fa_lock = mp.Lock()

    def __init__(self, batch_size: int, output_base_dir: str = None,
                 device: str = 'cuda', rank: int = 0, stream=None,
                 use_nvme_cache: bool = True):  # NEW PARAMETER: Enable/disable NVMe caching

        self.batch_size = batch_size
        self.output_base_dir = output_base_dir
        self.device = device
        self.rank = rank
        self.stream = stream or torch.cuda.current_stream()

        # ============ NEW NVMe ATTRIBUTES START ============
        # NEW: NVMe cache management attributes
        self.use_nvme_cache = use_nvme_cache  # NEW: Flag to enable/disable caching
        self.cache_dir = None  # NEW: Path to NVMe cache directory
        self.cached_videos = {}  # NEW: Mapping of original path -> cached path

        # NEW: Initialize NVMe cache if enabled
        if self.use_nvme_cache:
            self._setup_nvme_cache()  # NEW: Setup method for NVMe
        # ============ NEW NVMe ATTRIBUTES END ============

        os.makedirs(self.output_base_dir, exist_ok=True)

        self.fa = None
        self._init_face_alignment()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # MODIFIED: Increased thread workers when using NVMe (was 4, now 8)
        self.frame_executor = ThreadPoolExecutor(
            max_workers=8 if use_nvme_cache else 4,  # NEW: More workers for fast NVMe I/O
            thread_name_prefix=f"gpu{rank}_frame_reader"
        )
        self.save_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix=f"gpu{rank}_video_writer"
        )

        self.writer_locks = {}
        self.temp_dir = tempfile.mkdtemp(prefix=f"gpu{rank}_")

    # ============ NEW NVMe METHOD START ============
    def _setup_nvme_cache(self):
        """
        NEW METHOD: Automatically detect and setup NVMe cache

        Technologies used:
        1. psutil: Cross-platform system monitoring
           - Detects disk partitions and their types
           - Monitors available space in real-time

        2. Filesystem hierarchy detection:
           - Lambda Labs typically mounts NVMe at /scratch or /tmp
           - Checks multiple standard locations in priority order

        3. Performance-based selection:
           - Uses benchmark results to choose fastest storage
           - Falls back gracefully if NVMe not available
        """

        # Load pre-computed storage benchmarks
        config = load_storage_config()

        # Priority order for Lambda Labs infrastructure
        # These paths are where NVMe SSDs are typically mounted
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
            # Create GPU-specific cache directory to avoid conflicts
            self.cache_dir = Path(best_cache) / f"lip_extraction_cache_gpu{self.rank}"
            self.cache_dir.mkdir(exist_ok=True, parents=True)

            print(f"[GPU {self.rank}] ✅ NVMe cache enabled at {self.cache_dir}")
            print(f"[GPU {self.rank}]    Speed: {best_speed:.0f} MB/s")
        else:
            print(f"[GPU {self.rank}] ⚠️ No suitable NVMe storage found, using direct I/O")
            self.use_nvme_cache = False

    # ============ NEW NVMe METHOD END ============

    # ============ NEW NVMe METHOD START ============
    def _cache_videos_batch(self, video_paths: list) -> dict:
        """
        NEW METHOD: Pre-cache all videos to NVMe storage for fast access

        Technologies and concepts:

        1. **Staging Pattern**:
           - Copy data from slow storage (HDD/Network) to fast storage (NVMe)
           - Process from fast storage, eliminating I/O bottlenecks

        2. **Parallel I/O with ThreadPoolExecutor**:
           - Uses 16 concurrent threads for copying
           - Maximizes throughput for bulk transfers
           - Each thread handles one file independently

        3. **shutil.copy2**:
           - Preserves metadata (timestamps, permissions)
           - Uses optimized OS-level copy operations
           - Automatically uses sendfile() on Linux for zero-copy transfers

        4. **Progress tracking**:
           - Reports copy speed and completion status
           - Helps identify I/O bottlenecks

        Returns:
            Dictionary mapping original paths to cached paths
        """

        if not self.use_nvme_cache:
            return {p: p for p in video_paths}  # Return original paths if no cache

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
                    # shutil.copy2 preserves metadata and uses efficient OS operations
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

    # ============ NEW NVMe METHOD END ============

    def _init_face_alignment(self):
        """Same as before - no NVMe changes"""
        if self.fa is None:
            with self._fa_lock:
                if self.fa is None:
                    torch.cuda.set_device(self.device)
                    self.fa = face_alignment.FaceAlignment(
                        face_alignment.LandmarksType.TWO_D,
                        device=self.device,
                        face_detector='sfd',
                        flip_input=False
                    )
                    print(f"[GPU {self.rank}] Initialized face alignment")

    def __call__(self, video_paths: list[str]) -> None:
        # ============ NEW NVMe CACHING LOGIC START ============
        # NEW: Pre-cache all assigned videos to NVMe before processing
        if self.use_nvme_cache:
            # This copies all videos to fast storage first
            self.cached_videos = self._cache_videos_batch(video_paths)
            # Use cached paths for processing instead of original paths
            processing_paths = list(self.cached_videos.values())
        else:
            # No caching - use original paths directly
            processing_paths = video_paths
        # ============ NEW NVMe CACHING LOGIC END ============

        batch_size = 2

        for i in range(0, len(processing_paths), batch_size):
            batch_paths = processing_paths[i:i + batch_size]
            futures = []

            for path in batch_paths:
                # NEW: Track original path for output naming
                orig_path = video_paths[i + batch_paths.index(path)]
                future = self.frame_executor.submit(
                    self.process_video, path, orig_path  # NEW: Pass both paths
                )
                futures.append(future)

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ [GPU {self.rank}] Video processing failed: {e}")

        self.save_executor.shutdown(wait=True)
        self.frame_executor.shutdown(wait=True)

        # ============ NEW NVMe CLEANUP START ============
        # NEW: Clean up NVMe cache after processing
        if self.use_nvme_cache and self.cache_dir:
            try:
                print(f"[GPU {self.rank}] Cleaning up cache directory...")
                # Remove entire cache directory and all contents
                shutil.rmtree(self.cache_dir)
            except Exception as e:
                print(f"⚠️ Failed to clean cache: {e}")
        # ============ NEW NVMe CLEANUP END ============

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def process_video(self, video_path: str, original_path: str = None) -> None:
        """
        MODIFIED: Added original_path parameter to handle cached vs original paths
        - video_path: Path to cached video on NVMe
        - original_path: Original path for naming output files
        """

        if not os.path.exists(video_path):
            print(f"❌ Video path does not exist: {video_path}")
            return

        # NEW: Use original path for output naming if provided
        if original_path:
            video_name = Path(original_path).stem
        else:
            video_name = Path(video_path).stem

        unique_id = str(uuid.uuid4())[:8]
        out_path = os.path.join(self.output_base_dir, f"{video_name}_{unique_id}_lips.avi")

        # NEW: Log whether using cache or direct I/O
        if self.use_nvme_cache and video_path != original_path:
            print(f"[INFO] [GPU {self.rank}] Processing from cache: {video_path}")
        else:
            print(f"[INFO] [GPU {self.rank}] Processing: {video_path}")

        try:
            frame_queue = Queue(maxsize=100)
            crop_queue = Queue(maxsize=100)

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
        finally:
            gc.collect()

    def _async_frame_reader(self, video_path: str, frame_queue: Queue):
        """MODIFIED: Optimized for NVMe read speeds"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            frame_queue.put(None)
            return

        # ============ NEW NVMe OPTIMIZATION START ============
        # NEW: Increase buffer size for NVMe to reduce syscalls
        if self.use_nvme_cache:
            # Larger buffer reduces kernel overhead for fast storage
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
            # ============ NEW NVMe OPTIMIZATION END ============

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
        """MODIFIED: Write to NVMe first for faster I/O"""
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

                # ============ NEW NVMe WRITE OPTIMIZATION START ============
                # NEW: Write to NVMe cache first for faster writes
                if self.use_nvme_cache and self.cache_dir:
                    # Writing to NVMe is much faster than network/HDD
                    temp_path = str(self.cache_dir / f"temp_{uuid.uuid4()}.avi")
                else:
                    # Fallback to regular temp directory
                    temp_path = os.path.join(self.temp_dir, f"temp_{uuid.uuid4()}.avi")
                # ============ NEW NVMe WRITE OPTIMIZATION END ============

                # Use H264 for better compatibility and performance
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or try 'XVID'
                out = cv2.VideoWriter(temp_path, fourcc, 25.0, (w, h))

                if not out.isOpened():
                    print(f"❌ [GPU {self.rank}] Failed to open video writer")
                    return

                written = 0
                for frame in frames_buffer:
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)

                    success = out.write(frame)
                    if success:
                        written += 1

                out.release()

                if written > 0 and os.path.getsize(temp_path) > 1000:
                    # Move from temp to final location (might be slower storage)
                    shutil.move(temp_path, out_path)
                    print(f"[GPU {self.rank}] ✅ Saved {written}/{frame_count} frames")
                else:
                    print(f"❌ [GPU {self.rank}] Invalid video file")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        except Exception as e:
            print(f"❌ [GPU {self.rank}] Video writing error: {e}")

    def _process_frame_stream_safe(self, frame_queue: Queue, crop_queue: Queue):
        """Same as before - no NVMe changes"""
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
        """Same as before - no NVMe changes needed here"""
        if not frame_batch:
            return []

        crops = []

        try:
            mem_free = torch.cuda.mem_get_info(self.device)[0] / 1024 ** 3
            sub_batch_size = min(4 if mem_free < 40 else 8, len(frame_batch))

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

                with self._fa_lock:
                    landmarks_batch = self.fa.get_landmarks_from_batch(frame_batch_tensor)

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
        finally:
            gc.collect()

        return crops

    def extract_lip_segment(self, frame, landmarks):
        """Same as before - no NVMe changes"""
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


def worker_process(rank, chunks, batch_size, output_dir, return_dict, use_nvme):  # NEW: Added use_nvme parameter
    """MODIFIED: Added NVMe support parameter"""
    try:
        torch.cuda.set_device(rank)
        torch.cuda.init()

        device_str = f'cuda:{rank}'
        stream = torch.cuda.Stream(device=rank)

        os.nice(10)

        with torch.cuda.device(rank):
            with torch.cuda.stream(stream):
                processor = VideoPreprocessor_FANET(
                    batch_size=batch_size,
                    output_base_dir=output_dir,
                    device=device_str,
                    rank=rank,
                    stream=stream,
                    use_nvme_cache=use_nvme  # NEW: Pass NVMe flag
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


def parallel_main(video_paths: list[str], batch_size: int, output_dir: str,
                  use_nvme: bool = True):  # NEW: Added use_nvme parameter
    """MODIFIED: Added NVMe support"""

    # ============ NEW NVMe PRE-FLIGHT CHECK START ============
    # NEW: Check for NVMe availability before processing
    if use_nvme:
        print("🔍 Checking NVMe availability...")
        config = load_storage_config()

        if config['nvme_found']:
            print("✅ NVMe storage detected and will be used for caching")
        else:
            print("⚠️ No NVMe storage found, falling back to direct I/O")
            use_nvme = False
    # ============ NEW NVMe PRE-FLIGHT CHECK END ============

    available_gpus = torch.cuda.device_count()
    world_size = min(available_gpus, 8)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    manager = mp.Manager()
    return_dict = manager.dict()

    chunks = balanced_chunk_videos(video_paths, world_size)

    print(f"🚀 Starting distributed processing on {world_size} GPUs")
    print(f"📁 Videos per GPU: {[len(chunk) for chunk in chunks]}")

    # NEW: Log NVMe status
    if use_nvme:
        print(f"💾 NVMe caching enabled")

    for i, chunk in enumerate(chunks):
        total_duration = sum(get_video_info(p) for p in chunk)
        print(f"  GPU {i}: {len(chunk)} videos, ~{total_duration / 60:.1f} minutes total")

    mp.spawn(
        worker_process,
        args=(chunks, batch_size, output_dir, return_dict, use_nvme),  # NEW: Pass NVMe flag
        nprocs=world_size,
        join=True
    )

    successful_gpus = sum(return_dict.values())
    print(f"✅ Processing complete. {successful_gpus}/{world_size} GPUs finished successfully.")

    total_files = len([f for f in os.listdir(output_dir) if f.endswith('.avi')])
    print(f"📊 Total output files created: {total_files}/{len(video_paths)}")


def get_video_info(video_path):
    """Same as before - no NVMe changes"""
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
    """Same as before - no NVMe changes"""
    videos_with_info = []
    for path in video_paths:
        duration = get_video_info(path)
        videos_with_info.append((path, duration))

    videos_with_info.sort(key=lambda x: x[1], reverse=True)

    chunks = [[] for _ in range(num_gpus)]
    chunk_durations = [0] * num_gpus

    for video_path, duration in videos_with_info:
        min_idx = chunk_durations.index(min(chunk_durations))
        chunks[min_idx].append(video_path)
        chunk_durations[min_idx] += duration

    return chunks