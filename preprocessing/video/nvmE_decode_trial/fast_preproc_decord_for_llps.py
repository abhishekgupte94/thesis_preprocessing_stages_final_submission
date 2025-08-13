# fast_preproc_decord_mouth.py
import time, subprocess, torch
import torch.nn.functional as F
import face_alignment
from decord import VideoReader, gpu

# --- NVENC writer ---
def start_ffmpeg_nvenc_writer(out_path, width, height, fps):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps}",
        "-i", "-",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "vbr", "-b:v", "6M",
        "-movflags", "+faststart",
        out_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# --- Mouth-only boxes (48..67) + margin (matches your original ROI approach) ---
def landmarks_to_mouth_boxes(landmarks, margin=0.35):
    MOUTH = slice(48, 68)
    boxes = []
    for lm in landmarks:
        if lm is None or len(lm) == 0:
            boxes.append(None); continue
        pts = lm[0][MOUTH]
        x1, y1 = pts[:,0].min(), pts[:,1].min()
        x2, y2 = pts[:,0].max(), pts[:,1].max()
        w, h = (x2-x1), (y2-y1)
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        w2, h2 = w*(1+margin), h*(1+margin)
        boxes.append([cx-w2/2, cy-h2/2, cx+w2/2, cy+h2/2])
    return boxes

# --- Batched GPU crop+resize ---
def crop_resize_gpu(frames_rgb_t, boxes_xyxy, out_size=(224,224)):
    B, H, W, _ = frames_rgb_t.shape
    th, tw = out_size
    idxs = [i for i,b in enumerate(boxes_xyxy) if b is not None]
    if not idxs: return None, [], 0
    boxes = torch.tensor([boxes_xyxy[i] for i in idxs], device=frames_rgb_t.device, dtype=torch.float32)
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    x1.clamp_(0,W-1); x2.clamp_(0,W-1); y1.clamp_(0,H-1); y2.clamp_(0,H-1)

    xs = torch.linspace(-1,1,steps=tw,device=frames_rgb_t.device)
    ys = torch.linspace(-1,1,steps=th,device=frames_rgb_t.device)
    gy,gx = torch.meshgrid(ys,xs,indexing='ij')

    src_x = (gx.unsqueeze(0)+1)*(x2-x1).unsqueeze(1).unsqueeze(2)/2 + x1.unsqueeze(1).unsqueeze(2)
    src_y = (gy.unsqueeze(0)+1)*(y2-y1).unsqueeze(1).unsqueeze(2)/2 + y1.unsqueeze(1).unsqueeze(2)
    grid  = torch.stack(((src_x/(W-1))*2-1, (src_y/(H-1))*2-1), dim=-1)

    f_sel = frames_rgb_t[idxs].permute(0,3,1,2).float() / 255.0
    crops = F.grid_sample(f_sel, grid, mode='bilinear', align_corners=True)
    return crops, idxs, len(idxs)

@torch.inference_mode()
def process_video_gpu_lips(
    video_path: str,
    out_path: str,
    device: str = "cuda:0",
    out_size=(224,224),
    batch_frames: int = 64,
    mouth_margin: float = 0.35
):
    """
    Returns dict with timing and counts. Raises on hard failures.
    """
    dev = torch.device(device)
    t0 = time.perf_counter()

    # Open video with NVDEC on the *visible* GPU (when sharded via CUDA_VISIBLE_DEVICES)
    vr  = VideoReader(video_path, ctx=gpu(0))
    fps = float(vr.get_avg_fps()) or 25.0
    total_frames = len(vr)

    # Face-alignment on GPU
    fa  = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=str(dev), face_detector='sfd', flip_input=False
    )

    ff  = start_ffmpeg_nvenc_writer(out_path, out_size[1], out_size[0], fps)

    written_frames = 0
    idx = 0
    try:
        while idx < total_frames:
            batch_idx       = list(range(idx, min(idx+batch_frames, total_frames)))
            frames_rgb      = vr.get_batch(batch_idx)                                # (B,H,W,3) RGB uint8 on GPU
            frames_rgb_t    = torch.utils.dlpack.from_dlpack(frames_rgb.to_dlpack())
            fa_in           = frames_rgb_t.permute(0,3,1,2).float()/255.0           # (B,3,H,W)
            preds           = fa.get_landmarks_from_batch(fa_in)                    # list length B
            boxes           = landmarks_to_mouth_boxes(preds, margin=mouth_margin)
            crops, used, n  = crop_resize_gpu(frames_rgb_t, boxes, out_size=out_size)
            if n > 0:
                # NVENC expects BGR uint8
                bgr = (crops[:,[2,1,0],:,:]*255.0).clamp(0,255).byte()
                for k in range(bgr.shape[0]):
                    ff.stdin.write(bgr[k].permute(1,2,0).contiguous().cpu().numpy().tobytes())
                written_frames += bgr.shape[0]
            idx = batch_idx[-1] + 1
    except Exception:
        # make sure to close ffmpeg to avoid zombie processes, then re-raise
        try:
            ff.stdin.close()
            ff.wait(timeout=5)
        except Exception:
            ff.kill()
        raise

    # finalize ffmpeg and check status
    ff.stdin.close()
    rc = ff.wait()
    if rc != 0:
        err = ff.stderr.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg nvenc failed (rc={rc}). stderr:\n{err}")

    elapsed = time.perf_counter() - t0
    return {
        "video": video_path,
        "out": out_path,
        "elapsed_sec": round(elapsed, 3),
        "frames_total": int(total_frames),
        "frames_written": int(written_frames),
        "fps_effective": round((written_frames / elapsed) if elapsed > 0 else 0.0, 2),
    }
