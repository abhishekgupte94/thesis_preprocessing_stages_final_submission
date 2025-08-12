import os
import cv2
import subprocess
import numpy as np

W, H, FPS, FRAMES = 320, 240, 25, 60

def ffprobe_summary(path):
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,codec_long_name,profile,pix_fmt",
            "-of", "default=noprint_wrappers=1", path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
        return out.strip()
    except Exception as e:
        return f"(ffprobe unavailable or failed: {e})"

def main():
    out_path = "000024.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    if not writer.isOpened():
        print("❌ FAILED to open VideoWriter with mp4v")
        return

    for i in range(FRAMES):
        frame = (np.random.rand(H, W, 3) * 255).astype("uint8")
        cv2.putText(frame, f"mp4v frame={i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(f"✅ Wrote {out_path} ({os.path.getsize(out_path)} bytes)")

    print("ffprobe info:")
    print(ffprobe_summary(out_path))

if __name__ == "__main__":
    main()
