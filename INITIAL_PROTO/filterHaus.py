import cv2
from ultralytics import YOLO
import numpy as np
import torch

from dataclasses import dataclass
from typing import List

from pydub import AudioSegment
from scipy import signal
import sounddevice as sd

# =========================
# Channel "struct"
# =========================

@dataclass
class Channel:
    song_index: int               # index into songs list
    songs: List[str]              # list of file paths
    time_stamp: float = 0.0       # not used for seeking in this demo
    filter_type: int = 0          # 0 = none, 1 = low_pass
    channel_id: int = 0           # id for this channel
    play_pause: bool = False      # True = playing, False = paused
    started: bool = False         # has playback started at least once?
    intensity: int = 0            # 0–100 current filter intensity


FILTER_NONE = 0
FILTER_LOW_PASS = 1

# =========================
# Global audio state
# =========================

audio_samples = None      # np.ndarray, shape (N, channels), float32 in [-1,1]
audio_fs = None           # sample rate
audio_pos = 0             # current sample index
audio_playback_paused = True

# filter state
current_intensity = 0
last_filter_intensity = None
b = None
a = None
zi = None   # filter state per channel

audio_stream = None


def load_song_to_array(path: str):
    """
    Load an audio file via pydub and convert to a numpy float32 array.
    """
    seg = AudioSegment.from_file(path)
    fs = seg.frame_rate
    channels = seg.channels
    sample_width = seg.sample_width  # bytes per sample

    # Map sample width to numpy dtype
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    dtype = dtype_map[sample_width]
    raw = np.frombuffer(seg.raw_data, dtype=dtype)

    # Reshape to (N, channels)
    samples = raw.reshape((-1, channels)).astype(np.float32)

    # Normalize to [-1, 1]
    max_val = float(np.iinfo(dtype).max)
    samples /= max_val

    return samples, fs


def design_lowpass_filter(fs: float, channels: int):
    """
    Design / update low-pass filter coefficients based on global current_intensity.
    Sets globals b, a, zi.
    """
    global b, a, zi, last_filter_intensity, current_intensity

    # Only redesign if intensity changed
    if last_filter_intensity == current_intensity and b is not None:
        return

    t = current_intensity / 100.0  # 0 → no filter (high cutoff), 1 → strong filter (low cutoff)

    # Map intensity to cutoff frequency
    fc_max = 8000.0  # Hz
    fc_min = 400.0   # Hz
    fc = fc_max - t * (fc_max - fc_min)   # higher intensity → lower cutoff

    # Normalized cutoff for butter
    w = fc / (fs / 2.0)
    w = min(max(w, 0.001), 0.999)  # safety clamp

    b, a = signal.butter(2, w, btype='low')

    # Initial filter state for streaming
    zi_base = signal.lfilter_zi(b, a)          # shape (order,)
    zi = np.tile(zi_base.reshape(-1, 1), (1, channels)).astype(np.float32)

    last_filter_intensity = current_intensity
    print(f"[DSP] Redesigned low-pass: intensity={current_intensity}, cutoff≈{fc:.1f} Hz")


def audio_callback(outdata, frames, time_info, status):
    """
    sounddevice callback: reads next chunk, applies filter, writes to outdata.
    """
    global audio_samples, audio_fs, audio_pos
    global audio_playback_paused, b, a, zi

    if status:
        print("[AUDIO STATUS]", status)

    if audio_samples is None or audio_playback_paused:
        # Output silence if nothing loaded or paused
        outdata[:] = 0
        return

    num_samples = audio_samples.shape[0]
    num_channels = audio_samples.shape[1]

    # Get next chunk (loop if needed)
    start = audio_pos
    end = start + frames

    if end <= num_samples:
        chunk = audio_samples[start:end]
        audio_pos = end
    else:
        # Wrap-around looping
        part1 = audio_samples[start:]
        remaining = frames - (num_samples - start)
        part2 = audio_samples[:remaining]
        chunk = np.vstack((part1, part2))
        audio_pos = remaining

    # Apply filter if configured
    if b is not None and a is not None:
        design_lowpass_filter(audio_fs, num_channels)  # update coefficients if intensity changed

        # filtered output
        y = np.zeros_like(chunk)
        for ch in range(num_channels):
            y[:, ch], zi[:, ch] = signal.lfilter(b, a, chunk[:, ch], zi=zi[:, ch])
    else:
        y = chunk

    outdata[:] = y


def init_audio_system(song_path: str):
    """
    Load song into numpy array, set up sounddevice OutputStream with callback.
    """
    global audio_samples, audio_fs, audio_pos, audio_stream
    global b, a, zi, last_filter_intensity

    print(f"[AUDIO] Loading {song_path}")
    audio_samples, audio_fs = load_song_to_array(song_path)
    audio_pos = 0

    # Initialize filter state
    last_filter_intensity = None
    b = a = zi = None

    # Create output stream
    channels = audio_samples.shape[1]
    audio_stream = sd.OutputStream(
        samplerate=audio_fs,
        channels=channels,
        dtype='float32',
        callback=audio_callback
    )
    audio_stream.start()
    print(f"[AUDIO] Stream started at {audio_fs} Hz, channels={channels}")


# =========================
# Channel helpers
# =========================

def Load(song: int, songs: List[str], channel_id: int) -> Channel:
    """
    Initialize a Channel struct and the audio system for that song.
    """
    ch = Channel(song_index=song, songs=songs, channel_id=channel_id)
    init_audio_system(songs[song])
    print(f"[LOAD] Channel {channel_id} loaded with song index {song}")
    return ch


def Play(channel: Channel):
    """
    Toggle play/pause.
    """
    global audio_playback_paused

    if not channel.play_pause:
        # Resume / start
        print("[PLAY] Playing")
        audio_playback_paused = False
        channel.play_pause = True
        channel.started = True
    else:
        # Pause
        print("[PLAY] Paused")
        audio_playback_paused = True
        channel.play_pause = False


def Choose_Filter(filters: List[str], filter_index: int, channel: Channel):
    """
    Choose a filter from a list.
    For now: 0 = none, 1 = low_pass.
    """
    if filter_index < 0 or filter_index >= len(filters):
        raise ValueError("Invalid filter index")

    channel.filter_type = filter_index
    print(f"[FILTER] Channel {channel.channel_id} filter set to {filters[filter_index]}")

    # If switching to "none", clear filter
    global b, a, zi
    if filter_index == FILTER_NONE:
        b = a = zi = None


def Apply_filter(channel: Channel, intensity: int):
    """
    Update the intensity (0–100) that controls the real-time DSP.
    """
    global current_intensity

    intensity = max(0, min(100, intensity))
    channel.intensity = intensity
    current_intensity = intensity

    # If low-pass is active, the audio callback will pick this up
    if channel.filter_type == FILTER_LOW_PASS:
        # The actual filter design happens in the callback via design_lowpass_filter()
        pass
    else:
        # No filter means no DSP; callback just passes audio through
        pass


# =========================
# YOLO + CV
# =========================

objects = {
    0: "PERSON",
    1: "BICYCLE",
    2: "CAR",
    3: "MOTORCYCLE",
    4: "AIRPLANE",
    5: "BUS",
    6: "TRAIN",
    7: "TRUCK",
    8: "BOAT",
    9: "TRAFFIC LIGHT",
    10: "FIRE HYDRANT",
    11: "STOP SIGN",
    12: "PARKING METER",
    13: "BENCH",
    14: "BIRD",
    15: "CAT",
    16: "DOG",
    17: "HORSE",
    18: "SHEEP",
    19: "COW",
    20: "ELEPHANT",
    21: "BEAR",
    22: "ZEBRA",
    23: "GIRAFFE",
    24: "BACKPACK",
    25: "UMBRELLA",
    26: "HANDBAG",
    27: "TIE",
    28: "SUITCASE",
    29: "FRISBEE",
    30: "SKIS",
    31: "SNOWBOARD",
    32: "SPORTS BALL",
    33: "KITE",
    34: "BASEBALL BAT",
    35: "BASEBALL GLOVE",
    36: "SKATEBOARD",
    37: "SURFBOARD",
    38: "TENNIS RACKET",
    39: "BOTTLE",
    40: "WINE GLASS",
    41: "CUP",
    42: "FORK",
    43: "KNIFE",
    44: "SPOON",
    45: "BOWL",
    46: "BANANA",
    47: "APPLE",
    48: "SANDWICH",
    49: "ORANGE",
    50: "BROCCOLI",
    51: "CARROT",
    52: "HOT DOG",
    53: "PIZZA",
    54: "DONUT",
    55: "CAKE",
    56: "CHAIR",
    57: "COUCH",
    58: "POTTED PLANT",
    59: "BED",
    60: "DINING TABLE",
    61: "TOILET",
    62: "TV",
    63: "LAPTOP",
    64: "MOUSE",
    65: "REMOTE",
    66: "KEYBOARD",
    67: "CELL PHONE",
    68: "MICROWAVE",
    69: "OVEN",
    70: "TOASTER",
    71: "SINK",
    72: "REFRIGERATOR",
    73: "BOOK",
    74: "CLOCK",
    75: "VASE",
    76: "SCISSORS",
    77: "TEDDY BEAR",
    78: "HAIR DRIER",
    79: "TOOTHBRUSH"
}

print("MPS available:", torch.backends.mps.is_available())
model = YOLO("yolov8m-seg.pt")

# =========================
# Console inputs: songs + channel
# =========================

num_songs = int(input("How many songs do you want to load? "))
songs = []
for i in range(num_songs):
    path = input(f"Path for song {i} (e.g. ./songs/song{i+1}.mp3): ")
    songs.append(path)

song_index = int(input(f"Choose initial song index (0 to {num_songs - 1}): "))
channel_id = int(input("Channel id (e.g. 0): "))

channel = Load(song_index, songs, channel_id)

filters = ["none", "low_pass"]
print("Available filters:")
for i, f in enumerate(filters):
    print(f"  {i}: {f}")

filter_index = int(input("Choose filter index (0 = none, 1 = low_pass): "))
Choose_Filter(filters, filter_index, channel)

# Start playing
Play(channel)

# =========================
# Open camera
# =========================

videoCapture = cv2.VideoCapture(0)
if not videoCapture.isOpened():
    print("ERROR: Could not open camera.")
    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()
    sd.stop()
    raise SystemExit

print("Camera opened successfully. SPACE: play/pause, q: quit")

# =========================
# Main loop
# =========================

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Run YOLO (MPS if available)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    results = model(frame, device=device)
    result = results[0]

    if len(result.boxes) > 0:
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float")
    else:
        bboxes = []
        classes = []
        scores = []

    person_intensities = []

    for bbox, cls, score in zip(bboxes, classes, scores):
        x, y, x2, y2 = bbox

        if x2 - x <= 0 or y2 - y <= 0:
            continue

        # Draw bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
        label = f"{objects.get(int(cls), 'UNK')}: {score:.2f}"
        cv2.putText(frame, label, (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

        # Compute intensity only for PERSON
        if int(cls) == 0:
            center_y = (y + y2) / 2.0
            # bottom → 0, top → 100
            norm = 1.0 - (center_y / float(h))
            intensity = int(np.clip(norm * 100.0, 0, 100))
            person_intensities.append(intensity)

    max_intensity = max(person_intensities) if person_intensities else 0

    # Update audio filter intensity
    Apply_filter(channel, max_intensity)

    # Display intensity
    cv2.putText(frame, f"Intensity: {max_intensity}",
                (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Img", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        Play(channel)
    if key == ord('q'):
        break

# Cleanup
videoCapture.release()
cv2.destroyAllWindows()

if audio_stream is not None:
    audio_stream.stop()
    audio_stream.close()
sd.stop()
