"""
Real-time audio FX player + YOLOv8 vision-controlled intensity (pinch distance).

- Filters are chosen via terminal commands.
- Intensity (0.0–1.0) is controlled by YOLO based on the distance between
  the thumb tip and index finger tip:

    * Thumb & index together  -> intensity ~ 0.0
    * Thumb & index far apart -> intensity ~ 1.0 (clamped)

Additionally:
- NO FILTER is applied at all until a hand is detected.
  (Audio is dry when no hand is in frame.)

Press 'q' in the video window or type 'quit' in the terminal to stop.
"""

import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal

import cv2
from ultralytics import YOLO
import torch

# -----------------------------
# Config
# -----------------------------
AUDIO_FILE = "./songs/song10.mp3"
BLOCKSIZE = 1024

VIDEO_SOURCE = 0
YOLO_MODEL_PATH = "yolov8n-pose.pt"
PERSON_CLASS_ID = 0  # unused for hand model, but kept for completeness

CONF_THRESH = 0.4
VISION_SMOOTH_ALPHA = 0.2

# Global shared state
current_filter = "none"
current_intensity = 0.0
running = True
person_present = False   # now means: "hand with valid thumb/index detected"

fs = 44100
data = None
frame_idx = 0

# -----------------------------
# Filter design (HP / LP)
# -----------------------------
_last_filter_type = None
_last_intensity = None
_last_taps = None


def design_simple_fir(filter_type: str, intensity: float, fs: float, numtaps: int = 101):
    global _last_filter_type, _last_intensity, _last_taps

    intensity_key = round(float(intensity), 2)

    if filter_type == _last_filter_type and intensity_key == _last_intensity:
        return _last_taps

    nyq = fs / 2.0
    min_cut = 200.0
    max_cut = nyq - 500.0
    if max_cut <= min_cut:
        max_cut = nyq * 0.8

    cutoff = min_cut + intensity_key * (max_cut - min_cut)

    if filter_type == "lowpass":
        taps = signal.firwin(numtaps, cutoff / nyq)
    elif filter_type == "highpass":
        taps = signal.firwin(numtaps, cutoff / nyq, pass_zero=False)
    else:
        taps = None

    _last_filter_type = filter_type
    _last_intensity = intensity_key
    _last_taps = taps
    return taps


# -----------------------------
# Effect state + processing
# -----------------------------
MAX_DELAY_SEC = 1.0
MAX_FLANGER_DELAY_SEC = 0.02

echo_buffer = None
echo_pos = 0

delay_buffer = None
delay_pos = 0

flanger_buffer = None
flanger_pos = 0
flanger_phase = 0.0

reverb_buffer = None
reverb_pos = 0


def init_effect_buffers():
    global echo_buffer, delay_buffer, flanger_buffer, reverb_buffer
    global echo_pos, delay_pos, flanger_pos, reverb_pos, fs

    echo_buffer = np.zeros(int(fs * MAX_DELAY_SEC), dtype=np.float32)
    delay_buffer = np.zeros(int(fs * MAX_DELAY_SEC), dtype=np.float32)
    reverb_buffer = np.zeros(int(fs * MAX_DELAY_SEC), dtype=np.float32)
    flanger_buffer = np.zeros(int(fs * MAX_FLANGER_DELAY_SEC), dtype=np.float32)

    echo_pos = 0
    delay_pos = 0
    reverb_pos = 0
    flanger_pos = 0


def apply_effect(chunk: np.ndarray, effect: str, intensity: float) -> np.ndarray:
    global echo_buffer, echo_pos
    global delay_buffer, delay_pos
    global flanger_buffer, flanger_pos, flanger_phase
    global reverb_buffer, reverb_pos, fs

    if effect == "none":
        return chunk

    if effect in ("lowpass", "highpass"):
        taps = design_simple_fir(effect, intensity, fs)
        if taps is None:
            return chunk
        return signal.lfilter(taps, [1.0], chunk)

    if echo_buffer is None:
        init_effect_buffers()

    out = np.zeros_like(chunk)

    if effect == "echo":
        delay_sec = 0.2 + 0.5 * intensity
        delay_samp = max(1, min(int(fs * delay_sec), len(echo_buffer) - 1))
        feedback = 0.3 + 0.4 * intensity

        N = len(echo_buffer)
        pos = echo_pos

        for i, x in enumerate(chunk):
            read_idx = (pos - delay_samp) % N
            delayed = echo_buffer[read_idx]
            y = x + intensity * delayed
            out[i] = y
            echo_buffer[pos] = x + delayed * feedback
            pos = (pos + 1) % N

        echo_pos = pos
        return out

    if effect == "delay":
        delay_sec = 0.3 + 0.6 * intensity
        delay_samp = max(1, min(int(fs * delay_sec), len(delay_buffer) - 1))
        feedback = 0.5 + 0.4 * intensity
        wet_mix = 0.4 + 0.5 * intensity

        N = len(delay_buffer)
        pos = delay_pos

        for i, x in enumerate(chunk):
            read_idx = (pos - delay_samp) % N
            delayed = delay_buffer[read_idx]
            y = (1.0 - wet_mix) * x + wet_mix * delayed
            out[i] = y
            delay_buffer[pos] = x + delayed * feedback
            pos = (pos + 1) % N

        delay_pos = pos
        return out

    if effect == "flanger":
        rate = 0.1 + 0.9 * intensity
        max_delay = int(fs * MAX_FLANGER_DELAY_SEC)
        min_delay = 1
        depth = max_delay - min_delay
        N = len(flanger_buffer)
        pos = flanger_pos
        phase = flanger_phase

        for i, x in enumerate(chunk):
            lfo = (np.sin(2 * np.pi * phase) + 1.0) / 2.0
            delay_samp = int(min_delay + lfo * depth)

            read_idx = (pos - delay_samp) % N
            delayed = flanger_buffer[read_idx]

            y = x + intensity * delayed
            out[i] = y

            flanger_buffer[pos] = x + delayed * 0.7 * intensity

            pos = (pos + 1) % N
            phase += rate / fs

        flanger_pos = pos
        flanger_phase = phase
        return out

    if effect == "reverb":
        N = len(reverb_buffer)
        pos = reverb_pos

        d1 = max(1, int(0.03 * fs))
        d2 = max(1, int((0.08 + 0.12 * intensity) * fs))

        fb = 0.4 + 0.4 * intensity
        wet_mix = 0.3 + 0.5 * intensity

        for i, x in enumerate(chunk):
            r1 = reverb_buffer[(pos - d1) % N]
            r2 = reverb_buffer[(pos - d2) % N]
            reverb_sample = (r1 + r2) * 0.5
            y = (1.0 - wet_mix) * x + wet_mix * reverb_sample
            out[i] = y
            reverb_buffer[pos] = x + reverb_sample * fb
            pos = (pos + 1) % N

        reverb_pos = pos
        return out

    return chunk


# -----------------------------
# Audio callback
# -----------------------------
def audio_callback(outdata, frames, time_info, status):
    if status:
        print("Audio callback status:", status)

    global frame_idx, data, fs, current_filter, current_intensity, person_present

    start = frame_idx
    end = frame_idx + frames

    if end <= len(data):
        chunk = data[start:end]
        frame_idx = end
    else:
        part1 = data[start:]
        part2 = data[: (end - len(data))]
        chunk = np.concatenate((part1, part2), axis=0)
        frame_idx = end - len(data)

    # Only apply filter if a hand is present; otherwise play dry
    effect_to_use = current_filter if person_present else "none"
    processed = apply_effect(chunk, effect_to_use, current_intensity)

    processed = np.clip(processed, -1.0, 1.0)
    outdata[:] = processed.astype(np.float32).reshape(-1, 1)


# -----------------------------
# Control loop (terminal)
# -----------------------------
def control_loop():
    global current_filter, running

    print("\nControls:")
    print("  filter none")
    print("  filter lowpass")
    print("  filter highpass")
    print("  filter echo")
    print("  filter delay")
    print("  filter flanger")
    print("  filter reverb")
    print("  quit")
    print("\nIntensity + ON/OFF are controlled by YOLO (thumb–index pinch distance).\n")

    while running:
        try:
            cmd = input("> ").strip()
        except EOFError:
            break

        if not cmd:
            continue

        parts = cmd.split()
        if parts[0].lower() == "filter" and len(parts) >= 2:
            f = parts[1].lower()
            valid = ["none", "lowpass", "highpass", "echo", "delay", "flanger", "reverb"]
            if f in valid:
                current_filter = f
                print(f"Filter set to: {current_filter}")
            else:
                print("Unknown filter. Use:", ", ".join(valid))

        elif parts[0].lower() == "quit":
            print("Stopping...")
            running = False
            break

        else:
            print("Commands:")
            print("  filter none|lowpass|highpass|echo|delay|flanger|reverb")
            print("  quit")


# -----------------------------
# Vision setup + step (main thread)
# -----------------------------
def create_yolo_and_camera():
    print("Loading YOLO model:", YOLO_MODEL_PATH)
    model = YOLO(YOLO_MODEL_PATH)
    print("YOLO loaded.")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("ERROR: Could not open video source", VIDEO_SOURCE)
        return None, None

    return model, cap


def vision_step(model, cap, smoothed_intensity):
    """
    One iteration of the vision loop, using a hand keypoints pose model.

    - Looks for hand keypoints from the Ultralytics hand-keypoints dataset.
    - For each detected hand, uses the distance between thumb tip and
      index finger tip to control intensity:

        * Small distance (pinched) -> intensity ~ 0.0
        * Large distance           -> intensity ~ 1.0 (clamped)

    Uses the maximum pinch distance across all hands as the control signal.

    Returns (keep_running, new_smoothed_intensity).
    """
    global current_intensity, running, person_present

    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        return False, smoothed_intensity

    h, w, _ = frame.shape

    # Run the pose model
    results = model(frame, verbose=False)
    kp_obj = results[0].keypoints  # Ultralytics Keypoints object

    max_pinch_dist = None

    if kp_obj is not None and kp_obj.xy is not None:
        # kp_obj.xy shape: (num_hands, 21, 2) for hand-keypoints
        all_xy = kp_obj.xy

        thumb_idx = 4   # thumb tip
        index_idx = 8   # index finger tip

        for hand_xy in all_xy:
            hand_xy_np = hand_xy.cpu().numpy()

            if hand_xy_np.shape[0] <= max(thumb_idx, index_idx):
                continue

            thumb_x, thumb_y = hand_xy_np[thumb_idx]
            index_x, index_y = hand_xy_np[index_idx]

            # Euclidean distance between thumb tip and index tip
            dist = float(np.hypot(index_x - thumb_x, index_y - thumb_y))

            # Track maximum distance across all hands
            if max_pinch_dist is None or dist > max_pinch_dist:
                max_pinch_dist = dist

            # Draw fingertips and line (visual feedback)
            cv2.circle(frame, (int(thumb_x), int(thumb_y)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(index_x), int(index_y)), 5, (0, 0, 255), -1)
            cv2.line(
                frame,
                (int(thumb_x), int(thumb_y)),
                (int(index_x), int(index_y)),
                (255, 0, 0),
                2,
            )

    # Map pinch distance to audio intensity
    if max_pinch_dist is not None:
        person_present = True  # hand with valid thumb/index detected

        # Reference max distance: some fraction of the frame size
        # (so intensity reaches ~1 when fingers are nicely spread)
        ref_max = 0.4 * float(min(h, w))
        if ref_max <= 0:
            ref_max = 1.0

        vision_intensity = max_pinch_dist / ref_max
        # Clamp 0..1 (if distance exceeds ref_max, stays at 1)
        vision_intensity = max(0.0, min(1.0, vision_intensity))
    else:
        person_present = False
        vision_intensity = 0.0

    # Smooth to reduce jitter
    smoothed_intensity = (
        VISION_SMOOTH_ALPHA * vision_intensity +
        (1.0 - VISION_SMOOTH_ALPHA) * smoothed_intensity
    )
    current_intensity = smoothed_intensity

    pinch_str = f"{max_pinch_dist:.1f}px" if max_pinch_dist is not None else "N/A"
    text = f"Intensity: {current_intensity:.2f} | Pinch: {pinch_str} | Hand: {'YES' if person_present else 'NO'}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Hand Pinch → Audio Filter Intensity", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        return False, smoothed_intensity

    return True, smoothed_intensity


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print(f"Loading audio: {AUDIO_FILE}")
    raw, fs = sf.read(AUDIO_FILE, dtype="float32")

    if raw.ndim > 1:
        data = raw.mean(axis=1)
        print(f"Loaded {raw.shape[1]}-channel audio, mixed to mono.")
    else:
        data = raw
        print("Loaded mono audio.")

    init_effect_buffers()
    frame_idx = 0

    # Start control thread
    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    ctrl_thread.start()

    # YOLO + camera (on main thread)
    model, cap = create_yolo_and_camera()
    if model is None or cap is None:
        print("Vision setup failed, exiting.")
        running = False
    else:
        smoothed_intensity = 0.0

        # Start audio stream
        with sd.OutputStream(
            samplerate=fs,
            channels=1,
            callback=audio_callback,
            blocksize=BLOCKSIZE,
            dtype="float32",
        ):
            print("Playback started. Use terminal for filter changes, 'q' in video window or 'quit' in terminal to stop.")
            while running:
                ok, smoothed_intensity = vision_step(model, cap, smoothed_intensity)
                if not ok:
                    break
                sd.sleep(5)

        cap.release()
        cv2.destroyAllWindows()

    print("Shutting down...")
    running = False
    print("Exited cleanly.")
