"""
Real-time audio FX player.

Filters / effects:
  - none
  - lowpass
  - highpass
  - echo
  - delay
  - flanger
  - reverb

Controls (type in terminal while running):

  filter none
  filter lowpass
  filter highpass
  filter echo
  filter delay
  filter flanger
  filter reverb
  intensity 0.5      # 0.0–1.0
  quit
"""

import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal

# -----------------------------
# Config
# -----------------------------
AUDIO_FILE = "./songs/song10.mp3"  # change to your file
BLOCKSIZE = 1024

# Global shared state
current_filter = "none"   # "none", "lowpass", "highpass", "echo", "delay", "flanger", "reverb"
current_intensity = 0.5   # 0.0–1.0
running = True

# Will be set after loading
fs = 44100
data = None
frame_idx = 0

# -----------------------------
# Filter design (for HP / LP)
# -----------------------------
_last_filter_type = None
_last_intensity = None
_last_taps = None


def design_simple_fir(filter_type: str, intensity: float, fs: float, numtaps: int = 101):
    """
    Design lowpass or highpass FIR filter.

    intensity 0.0–1.0 maps to cutoff between min_cut and max_cut.
    """
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
# Delay / echo / flanger / reverb state
# -----------------------------

# We’ll preallocate big buffers and reuse them.
MAX_DELAY_SEC = 1.0          # 1 second max delay for echo/delay/reverb
MAX_FLANGER_DELAY_SEC = 0.02 # 20 ms max delay for flanger

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
    global echo_pos, delay_pos, flanger_pos, reverb_pos

    echo_buffer = np.zeros(int(fs * MAX_DELAY_SEC), dtype=np.float32)
    delay_buffer = np.zeros(int(fs * MAX_DELAY_SEC), dtype=np.float32)
    reverb_buffer = np.zeros(int(fs * MAX_DELAY_SEC), dtype=np.float32)
    flanger_buffer = np.zeros(int(fs * MAX_FLANGER_DELAY_SEC), dtype=np.float32)

    echo_pos = 0
    delay_pos = 0
    reverb_pos = 0
    flanger_pos = 0


# -----------------------------
# Effect processing
# -----------------------------

def apply_effect(chunk: np.ndarray, effect: str, intensity: float) -> np.ndarray:
    """
    chunk: 1D mono float32 array
    effect: current_filter name
    intensity: 0.0–1.0
    """
    global echo_buffer, echo_pos
    global delay_buffer, delay_pos
    global flanger_buffer, flanger_pos, flanger_phase
    global reverb_buffer, reverb_pos, fs

    if effect == "none":
        return chunk

    # ---- Lowpass / Highpass ----
    if effect in ("lowpass", "highpass"):
        taps = design_simple_fir(effect, intensity, fs)
        if taps is None:
            return chunk
        # lfilter with pre-existing state would be better, but for simplicity
        # we filter per block; for long songs this is ok-ish.
        return signal.lfilter(taps, [1.0], chunk)

    # Ensure buffers are initialized
    if echo_buffer is None:
        init_effect_buffers()

    out = np.zeros_like(chunk)

    # ---- Echo ----
    if effect == "echo":
        # delay time 0.2–0.7 s
        delay_sec = 0.2 + 0.5 * intensity
        delay_samp = max(1, min(int(fs * delay_sec), len(echo_buffer) - 1))
        feedback = 0.3 + 0.4 * intensity  # 0.3–0.7

        N = len(echo_buffer)
        pos = echo_pos

        for i, x in enumerate(chunk):
            read_idx = (pos - delay_samp) % N
            delayed = echo_buffer[read_idx]
            y = x + intensity * delayed
            out[i] = y
            # write new value into buffer (simple feedback echo)
            echo_buffer[pos] = x + delayed * feedback
            pos = (pos + 1) % N

        echo_pos = pos
        return out

    # ---- Delay (more separated repeats, drier signal) ----
    if effect == "delay":
        # delay time 0.3–0.9 s
        delay_sec = 0.3 + 0.6 * intensity
        delay_samp = max(1, min(int(fs * delay_sec), len(delay_buffer) - 1))
        feedback = 0.5 + 0.4 * intensity  # 0.5–0.9
        wet_mix = 0.4 + 0.5 * intensity   # more intensity = wetter

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

    # ---- Flanger ----
    if effect == "flanger":
        # LFO settings
        rate = 0.1 + 0.9 * intensity   # 0.1–1.0 Hz
        max_delay = int(fs * MAX_FLANGER_DELAY_SEC)
        min_delay = 1
        depth = max_delay - min_delay
        N = len(flanger_buffer)
        pos = flanger_pos
        phase = flanger_phase

        for i, x in enumerate(chunk):
            # LFO for delay in samples
            lfo = (np.sin(2 * np.pi * phase) + 1.0) / 2.0  # 0–1
            delay_samp = int(min_delay + lfo * depth)

            read_idx = (pos - delay_samp) % N
            delayed = flanger_buffer[read_idx]

            # classic flanger: mix dry & delayed
            y = x + intensity * delayed
            out[i] = y

            # write (feedback flanger)
            flanger_buffer[pos] = x + delayed * 0.7 * intensity

            pos = (pos + 1) % N
            phase += rate / fs

        flanger_pos = pos
        flanger_phase = phase
        return out

    # ---- Reverb (simple multi-tap feedback delay) ----
    if effect == "reverb":
        N = len(reverb_buffer)
        pos = reverb_pos

        # Two taps, scaled by intensity
        # Early reflection + late reflection
        d1 = max(1, int(0.03 * fs))  # 30 ms
        d2 = max(1, int((0.08 + 0.12 * intensity) * fs))  # 80–200 ms

        fb = 0.4 + 0.4 * intensity   # feedback
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

    # fallback
    return chunk


# -----------------------------
# Audio callback
# -----------------------------

def audio_callback(outdata, frames, time_info, status):
    if status:
        print("Audio callback status:", status)

    global frame_idx, data, fs, current_filter, current_intensity

    # loop audio
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

    # chunk is mono (1D)
    processed = apply_effect(chunk, current_filter, current_intensity)

    # clip & convert
    processed = np.clip(processed, -1.0, 1.0)
    outdata[:] = processed.astype(np.float32).reshape(-1, 1)


# -----------------------------
# Control loop (terminal commands)
# -----------------------------

def control_loop():
    global current_filter, current_intensity, running

    print("\nControls:")
    print("  filter none")
    print("  filter lowpass")
    print("  filter highpass")
    print("  filter echo")
    print("  filter delay")
    print("  filter flanger")
    print("  filter reverb")
    print("  intensity 0.3   # 0.0–1.0")
    print("  quit\n")

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

        elif parts[0].lower() == "intensity" and len(parts) >= 2:
            try:
                val = float(parts[1])
                val = max(0.0, min(1.0, val))
                current_intensity = val
                print(f"Intensity set to: {current_intensity:.2f}")
            except ValueError:
                print("Intensity must be a number between 0.0 and 1.0")

        elif parts[0].lower() == "quit":
            print("Stopping...")
            running = False
            break

        else:
            print("Commands:")
            print("  filter none|lowpass|highpass|echo|delay|flanger|reverb")
            print("  intensity <0.0–1.0>")
            print("  quit")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    print(f"Loading {AUDIO_FILE} ...")
    raw, fs = sf.read(AUDIO_FILE, dtype="float32")

    # mix to mono for simpler effect code
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

    # Start audio stream
    with sd.OutputStream(
        samplerate=fs,
        channels=1,
        callback=audio_callback,
        blocksize=BLOCKSIZE,
        dtype="float32",
    ):
        print("Playback started. Type commands in the terminal.")
        while running:
            sd.sleep(100)

    print("Exited cleanly.")
