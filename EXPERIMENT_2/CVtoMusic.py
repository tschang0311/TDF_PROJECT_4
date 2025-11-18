import cv2
from ultralytics import YOLO
import numpy as np
import torch
import pygame
import os
import glob
from scipy import signal

# =============================================================================
# Audio Setup (concurrent sounds via Channels)
# =============================================================================

# Initialize the Pygame mixer for audio playback (must be done before loading sounds)
pygame.mixer.init()

# =============================================================================
# High-Pass Filter Function
# =============================================================================

def apply_highpass_filter(sound_obj, cutoff_freq=1000, sample_rate=44100):
    """
    Apply a high-pass filter to a pygame Sound object.
    Returns a new Sound object with the filtered audio.
    """
    # Get audio array from sound object
    sound_array = pygame.sndarray.array(sound_obj)
    
    # Handle stereo vs mono
    if len(sound_array.shape) == 1:
        # Mono audio
        audio_data = sound_array.astype(np.float32)
        is_stereo = False
    else:
        # Stereo audio - process both channels
        audio_data = sound_array.astype(np.float32)
        is_stereo = True
    
    # Design high-pass Butterworth filter (4th order)
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    sos = signal.butter(4, normalized_cutoff, btype='high', output='sos')
    
    # Apply filter
    if is_stereo:
        filtered_audio = np.zeros_like(audio_data)
        for channel in range(audio_data.shape[1]):
            filtered_audio[:, channel] = signal.sosfilt(sos, audio_data[:, channel])
    else:
        filtered_audio = signal.sosfilt(sos, audio_data)
    
    # Convert back to int16 and create new Sound object
    filtered_audio = np.clip(filtered_audio, -32768, 32767).astype(np.int16)
    filtered_sound = pygame.sndarray.make_sound(filtered_audio)
    
    return filtered_sound

# =============================================================================
# Low-Pass Filter Function
# =============================================================================

def apply_lowpass_filter(sound_obj, cutoff_freq=1000, sample_rate=44100):
    """
    Apply a low-pass filter to a pygame Sound object.
    Returns a new Sound object with the filtered audio.
    """
    # Get audio array from sound object
    sound_array = pygame.sndarray.array(sound_obj)
    
    # Handle stereo vs mono
    if len(sound_array.shape) == 1:
        # Mono audio
        audio_data = sound_array.astype(np.float32)
        is_stereo = False
    else:
        # Stereo audio - process both channels
        audio_data = sound_array.astype(np.float32)
        is_stereo = True
    
    # Design low-pass Butterworth filter (4th order)
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    sos = signal.butter(4, normalized_cutoff, btype='low', output='sos')
    
    # Apply filter
    if is_stereo:
        filtered_audio = np.zeros_like(audio_data)
        for channel in range(audio_data.shape[1]):
            filtered_audio[:, channel] = signal.sosfilt(sos, audio_data[:, channel])
    else:
        filtered_audio = signal.sosfilt(sos, audio_data)
    
    # Convert back to int16 and create new Sound object
    filtered_audio = np.clip(filtered_audio, -32768, 32767).astype(np.int16)
    filtered_sound = pygame.sndarray.make_sound(filtered_audio)
    
    return filtered_sound

# Create Sound objects for each song; keep names descriptive and in camelCase
songOneSound = pygame.mixer.Sound("song.mp3")    # Played when >= 1 person is detected
songTwoSound = pygame.mixer.Sound("song2.mp3")   # Played when >= 2 people are detected
songThreeSound = pygame.mixer.Sound("song3.mp3") # Played when >= 3 people are detected

# Set base volumes for each song (0.0 to 1.0)
songOneSound.set_volume(0.8)
songTwoSound.set_volume(0.8)
songThreeSound.set_volume(0.8)

# Create high-pass filtered versions of each song (pre-processed for efficiency)
print("Creating high-pass filtered audio versions...")
songOneSoundFilteredHP = apply_highpass_filter(songOneSound)
songTwoSoundFilteredHP = apply_highpass_filter(songTwoSound)
songThreeSoundFilteredHP = apply_highpass_filter(songThreeSound)
songOneSoundFilteredHP.set_volume(0.8)
songTwoSoundFilteredHP.set_volume(0.8)
songThreeSoundFilteredHP.set_volume(0.8)
print("High-pass filtered audio versions created.")

# Create low-pass filtered versions of each song (pre-processed for efficiency)
print("Creating low-pass filtered audio versions...")
songOneSoundFilteredLP = apply_lowpass_filter(songOneSound)
songTwoSoundFilteredLP = apply_lowpass_filter(songTwoSound)
songThreeSoundFilteredLP = apply_lowpass_filter(songThreeSound)
songOneSoundFilteredLP.set_volume(0.8)
songTwoSoundFilteredLP.set_volume(0.8)
songThreeSoundFilteredLP.set_volume(0.8)
print("Low-pass filtered audio versions created.")

# Reserve three channels so all three songs can play concurrently without interrupting each other
channelOne = pygame.mixer.Channel(0)
channelTwo = pygame.mixer.Channel(1)
channelThree = pygame.mixer.Channel(2)

# =============================================================================
# Detection Parameters
# =============================================================================

# Minimum confidence score required to consider a "CELL PHONE" detection valid
cellPhoneScoreThreshold = 0.40

# Optional stability filter: require a stable cell phone count for N frames before changing audio
# Set to 0 for immediate reaction (no debounce)
stabilityFrameThreshold = 0  # Example: set to 3–5 to reduce rapid flicker of audio state

# Bookkeeping counters for stability filtering
stableFrameCount = 0             # Number of consecutive frames with the same cell phone count
lastCommittedCount = 0           # Cell phone count that has already been applied to audio
lastSeenCount = 0                # Most recent instantaneous cell phone count seen
lastCommittedFilterType = "none" # Last committed filter type: "none", "highpass", or "lowpass"

# =============================================================================
# GPU / Accelerator Info (Apple Silicon MPS)
# =============================================================================

# Print whether Apple's Metal Performance Shaders backend is available
print("MPS available:", torch.backends.mps.is_available())

# =============================================================================
# Load YOLOv8 Segmentation Model
# =============================================================================

# Instantiate a pretrained YOLO model; ensure 'yolov8m-seg.pt' is available locally
yoloModel = YOLO("yolov8m-seg.pt")

# =============================================================================
# COCO Dataset Label Map (ID → Name)
# =============================================================================

cocoClassMap = {
    0: "PERSON", 1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE", 4: "AIRPLANE", 5: "BUS",
    6: "TRAIN", 7: "TRUCK", 8: "BOAT", 9: "TRAFFIC LIGHT", 10: "FIRE HYDRANT",
    11: "STOP SIGN", 12: "PARKING METER", 13: "BENCH", 14: "BIRD", 15: "CAT",
    16: "DOG", 17: "HORSE", 18: "SHEEP", 19: "COW", 20: "ELEPHANT", 21: "BEAR",
    22: "ZEBRA", 23: "GIRAFFE", 24: "BACKPACK", 25: "UMBRELLA", 26: "HANDBAG",
    27: "TIE", 28: "SUITCASE", 29: "FRISBEE", 30: "SKIS", 31: "SNOWBOARD",
    32: "SPORTS BALL", 33: "KITE", 34: "BASEBALL BAT", 35: "BASEBALL GLOVE",
    36: "SKATEBOARD", 37: "SURFBOARD", 38: "TENNIS RACKET", 39: "BOTTLE",
    40: "WINE GLASS", 41: "CUP", 42: "FORK", 43: "KNIFE", 44: "SPOON",
    45: "BOWL", 46: "BANANA", 47: "APPLE", 48: "SANDWICH", 49: "ORANGE",
    50: "BROCCOLI", 51: "CARROT", 52: "HOT DOG", 53: "PIZZA", 54: "DONUT",
    55: "CAKE", 56: "CHAIR", 57: "COUCH", 58: "POTTED PLANT", 59: "BED",
    60: "DINING TABLE", 61: "TOILET", 62: "TV", 63: "LAPTOP", 64: "MOUSE",
    65: "REMOTE", 66: "KEYBOARD", 67: "CELL PHONE", 68: "MICROWAVE",
    69: "OVEN", 70: "TOASTER", 71: "SINK", 72: "REFRIGERATOR", 73: "BOOK",
    74: "CLOCK", 75: "VASE", 76: "SCISSORS", 77: "TEDDY BEAR", 78: "HAIR DRIER",
    79: "TOOTHBRUSH"
}

# =============================================================================
# Track Library Discovery
# =============================================================================

scriptDir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()  # Get script directory
availableTracks = sorted([os.path.basename(f) for f in glob.glob(os.path.join(scriptDir, "*.mp3"))])  # Find all MP3 files, extract names, sort alphabetically
print(f"Found {len(availableTracks)} tracks: {availableTracks}")  # Debug: print found tracks

# =============================================================================
# Video Source (webcam 0 by default)
# =============================================================================

# Open the default camera; change the index if you have multiple cameras
videoCapture = cv2.VideoCapture(0)

# Check if camera opened successfully
if not videoCapture.isOpened():
    print("ERROR: Could not open camera. Please check if camera is connected and not in use by another application.")
    exit(1)

print("Camera opened successfully")

# =============================================================================
# Audio Control Based on Cell Phone Count
# =============================================================================

def applyAudioForCellPhoneCount(cellPhoneCount: int, filterType: str = "none") -> None:
    """
    Keep channel playback aligned with the current cell phone count:
      >= 1 → channelOne loops songOneSound (or filtered version)
      >= 2 → channelTwo loops songTwoSound (or filtered version)
      >= 3 → channelThree loops songThreeSound (or filtered version)
      < threshold → stop corresponding higher-index channels
    
    Args:
        cellPhoneCount: Number of cell phones detected
        filterType: "none", "highpass", or "lowpass" to select filter type
    """
    # Select sound objects based on filter state
    if filterType == "highpass":
        sound1, sound2, sound3 = songOneSoundFilteredHP, songTwoSoundFilteredHP, songThreeSoundFilteredHP
    elif filterType == "lowpass":
        sound1, sound2, sound3 = songOneSoundFilteredLP, songTwoSoundFilteredLP, songThreeSoundFilteredLP
    else:
        sound1, sound2, sound3 = songOneSound, songTwoSound, songThreeSound
    
    # ---------------------------
    # Manage channelOne / songOne
    # ---------------------------
    if cellPhoneCount >= 1:
        # Check if we need to switch between filtered/unfiltered
        current_sound = channelOne.get_sound()
        if current_sound is None or current_sound != sound1:
            # Stop current and switch to correct version
            channelOne.stop()
            channelOne.play(sound1, loops=-1)
        elif not channelOne.get_busy():
            # Not playing, start it
            channelOne.play(sound1, loops=-1)
    else:
        # If currently playing but no longer needed, stop it
        if channelOne.get_busy():
            channelOne.stop()

    # ---------------------------
    # Manage channelTwo / songTwo
    # ---------------------------
    if cellPhoneCount >= 2:
        current_sound = channelTwo.get_sound()
        if current_sound is None or current_sound != sound2:
            channelTwo.stop()
            channelTwo.play(sound2, loops=-1)
        elif not channelTwo.get_busy():
            channelTwo.play(sound2, loops=-1)
    else:
        if channelTwo.get_busy():
            channelTwo.stop()

    # -----------------------------
    # Manage channelThree / songThree
    # -----------------------------
    if cellPhoneCount >= 3:
        current_sound = channelThree.get_sound()
        if current_sound is None or current_sound != sound3:
            channelThree.stop()
            channelThree.play(sound3, loops=-1)
        elif not channelThree.get_busy():
            channelThree.play(sound3, loops=-1)
    else:
        if channelThree.get_busy():
            channelThree.stop()

# =============================================================================
# Main Loop: Read Frames, Run YOLO, Draw Results, Control Audio
# =============================================================================

print("Starting main loop...")
try:
    while True:
        # Grab a frame from the camera
        didReadFrame, frameBgr = videoCapture.read()
        if not didReadFrame:
            # If frame read fails (camera unplugged, etc.), exit the loop gracefully
            print("Failed to read frame from camera")
            break

        # Run YOLO inference on the raw frame
        # - device="mps" uses Apple Metal if available; adjust to "cuda" or "cpu" as needed
        try:
            # Check if MPS is available, otherwise use CPU
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            inferenceResultsList = yoloModel(frameBgr, device=device)
        except Exception as e:
            print(f"ERROR during YOLO inference: {e}")
            break
        inferenceResult = inferenceResultsList[0]  # We only need the first (and only) result

        # Extract detection tensors and convert to CPU numpy arrays for easy handling
        # - xyxy: bounding boxes in (left, top, right, bottom)
        # - cls: class indices
        # - conf: confidence scores
        boundingBoxes = np.array(inferenceResult.boxes.xyxy.cpu(), dtype="int")
        classIndices = np.array(inferenceResult.boxes.cls.cpu(), dtype="int")
        confidenceScores = np.array(inferenceResult.boxes.conf.cpu(), dtype="float")

        # Get frame dimensions for quadrant detection
        frameHeight, frameWidth = frameBgr.shape[:2]
        midX = frameWidth // 2  # Vertical dividing line (left/right)
        midY = frameHeight // 2  # Horizontal dividing line (top/bottom)
        
        # Draw quadrant dividing lines for visual reference
        # Vertical line (divides left/right)
        cv2.line(frameBgr, (midX, 0), (midX, frameHeight), (128, 128, 128), 2)
        # Horizontal line (divides top/bottom)
        cv2.line(frameBgr, (0, midY), (frameWidth, midY), (128, 128, 128), 2)
        
        # Reset the per-frame count of valid CELL PHONE detections
        currentCellPhoneCount = 0
        # Track if any cell phone is in upper right quadrant
        cellPhoneInUpperRight = False
        # Track if any cell phone is in lower right quadrant
        cellPhoneInLowerRight = False

        # Iterate through all detections for this frame
        for boxCoordinates, classIndex, confidenceScore in zip(boundingBoxes, classIndices, confidenceScores):
            (left, top, right, bottom) = boxCoordinates  # Unpack box corners

            # Skip degenerate boxes that have invalid geometry
            if right - left <= 0 or bottom - top <= 0:
                continue

            # Prepare a human-readable label using the class name and confidence
            className = cocoClassMap.get(int(classIndex), "UNKNOWN").upper()
            
            # Only process and draw CELL PHONE detections
            if className == "CELL PHONE" and float(confidenceScore) >= cellPhoneScoreThreshold:
                # Draw a rectangle around the detection for visualization
                cv2.rectangle(frameBgr, (left, top), (right, bottom), (0, 0, 225), 2)
                
                labelText = f"{className}: {float(confidenceScore):.2f}"

                # Draw the label just above the bounding box
                cv2.putText(
                    frameBgr,
                    labelText,
                    (left, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 225),
                    2
                )
                
                currentCellPhoneCount += 1
                
                # Check if cell phone center is in upper right or lower right quadrant
                # Upper right: x > midX (right half) AND y < midY (upper half)
                # Lower right: x > midX (right half) AND y >= midY (lower half)
                centerX = (left + right) // 2
                centerY = (top + bottom) // 2
                if centerX > midX and centerY < midY:
                    cellPhoneInUpperRight = True
                elif centerX > midX and centerY >= midY:
                    cellPhoneInLowerRight = True

        # Highlight upper right quadrant if cell phone detected there
        if cellPhoneInUpperRight:
            # Draw a semi-transparent yellow overlay on upper right quadrant
            overlay = np.zeros((midY, frameWidth - midX, 3), dtype=np.uint8)
            overlay[:] = (0, 255, 255)  # Yellow color
            frameBgr[0:midY, midX:frameWidth] = cv2.addWeighted(
                frameBgr[0:midY, midX:frameWidth], 0.7, overlay, 0.3, 0
            )
            # Draw bright border around upper right quadrant
            cv2.rectangle(frameBgr, (midX, 0), (frameWidth, midY), (0, 255, 255), 4)
        
        # Highlight lower right quadrant if cell phone detected there
        if cellPhoneInLowerRight:
            # Draw a semi-transparent overlay on lower right quadrant
            overlay = np.zeros((frameHeight - midY, frameWidth - midX, 3), dtype=np.uint8)
            overlay[:] = (255, 0, 255)  # Magenta color
            frameBgr[midY:frameHeight, midX:frameWidth] = cv2.addWeighted(
                frameBgr[midY:frameHeight, midX:frameWidth], 0.7, overlay, 0.3, 0
            )
            # Draw bright border around lower right quadrant
            cv2.rectangle(frameBgr, (midX, midY), (frameWidth, frameHeight), (255, 0, 255), 4)
        
        # Label quadrants (draw after detection to show active state)
        cv2.putText(frameBgr, "UL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        urColor = (0, 255, 255) if cellPhoneInUpperRight else (255, 255, 0)
        cv2.putText(frameBgr, "UR", (frameWidth - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, urColor, 2)
        cv2.putText(frameBgr, "LL", (10, frameHeight - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        lrColor = (255, 0, 255) if cellPhoneInLowerRight else (128, 128, 128)
        cv2.putText(frameBgr, "LR", (frameWidth - 50, frameHeight - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lrColor, 2)

        # -----------------------------------------------------------------------------
        # Determine filter type based on quadrant
        # Lower right takes priority if both quadrants have phones (or just one active)
        # -----------------------------------------------------------------------------
        if cellPhoneInLowerRight:
            currentFilterType = "lowpass"
        elif cellPhoneInUpperRight:
            currentFilterType = "highpass"
        else:
            currentFilterType = "none"
        
        # -----------------------------------------------------------------------------
        # Optional stability debounce: require a stable count for N frames before commit
        # Note: Filter state changes are applied immediately for responsiveness
        # -----------------------------------------------------------------------------
        if stabilityFrameThreshold <= 0:
            # Immediate mode: apply the audio state every frame based on the current count
            applyAudioForCellPhoneCount(currentCellPhoneCount, filterType=currentFilterType)
            lastCommittedCount = currentCellPhoneCount
            lastCommittedFilterType = currentFilterType
        else:
            # Debounced mode: only change audio after N consecutive frames with same count
            # But apply filter changes immediately
            filterStateChanged = currentFilterType != lastCommittedFilterType
            
            if currentCellPhoneCount == lastSeenCount:
                stableFrameCount += 1  # Count another same-value frame
            else:
                # Count changed: reset stability counter and track the new value
                stableFrameCount = 0
                lastSeenCount = currentCellPhoneCount

            # Apply audio if filter state changed (immediate) or count is stable and changed
            shouldApplyAudio = filterStateChanged or (stableFrameCount >= stabilityFrameThreshold and currentCellPhoneCount != lastCommittedCount)
            
            if shouldApplyAudio:
                applyAudioForCellPhoneCount(currentCellPhoneCount, filterType=currentFilterType)
                lastCommittedCount = currentCellPhoneCount
                lastCommittedFilterType = currentFilterType
                if not filterStateChanged:
                    stableFrameCount = 0  # Reset so we can detect future changes cleanly

        # -----------------------------------------------------------------------------
        # Track Library Display: transparent overlay at top showing all available tracks
        # -----------------------------------------------------------------------------
        overlayHeight = 0  # Initialize overlay height
        if len(availableTracks) > 0:  # Only display if tracks are found
            textHeight = 30  # Vertical spacing per line
            padding = 10  # Horizontal/vertical padding
            overlayHeight = (len(availableTracks) + 1) * textHeight + padding * 2  # Calculate overlay height
            overlayHeight = min(overlayHeight, frameBgr.shape[0] // 2)  # Limit height to half frame
            overlay = np.zeros((overlayHeight, frameBgr.shape[1], 3), dtype=np.uint8)  # Create black overlay
            cv2.addWeighted(frameBgr[0:overlayHeight, :], 0.4, overlay, 0.6, 0, frameBgr[0:overlayHeight, :])  # Blend (60% black, 40% frame)
            cv2.putText(frameBgr, "Library", (padding, textHeight), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # Title
            for idx, trackName in enumerate(availableTracks):  # Draw each track name in bold
                yPos = (idx + 2) * textHeight + padding
                if yPos < overlayHeight - padding:  # Only draw if within overlay bounds
                    cv2.putText(frameBgr, trackName, (padding, yPos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # -----------------------------------------------------------------------------
        # On-screen status overlay: show the computed cell phone count and filter status
        # -----------------------------------------------------------------------------
        statusY = (overlayHeight + 30) if len(availableTracks) > 0 else 40  # Position below track list or at top if no tracks
        cv2.putText(
            frameBgr,
            f"CELL PHONE count: {currentCellPhoneCount}",
            (20, statusY),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Show filter status
        filterStatusY = statusY + 40
        if cellPhoneInLowerRight:
            filterColor = (255, 0, 255)  # Magenta
            filterText = "LOW-PASS FILTER: ACTIVE"
        elif cellPhoneInUpperRight:
            filterColor = (0, 255, 255)  # Yellow
            filterText = "HIGH-PASS FILTER: ACTIVE"
        else:
            filterColor = (128, 128, 128)  # Gray
            filterText = "FILTER: INACTIVE"
        cv2.putText(
            frameBgr,
            filterText,
            (20, filterStatusY),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            filterColor,
            2
        )

        # Display the annotated frame in a window
        cv2.imshow("YOLO Detection", frameBgr)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed 'q' to quit")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user (Ctrl+C)")
except Exception as e:
    print(f"ERROR in main loop: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Cleanup: release camera, close windows, stop any running audio
# =============================================================================

# Release the camera device
videoCapture.release()

# Close any OpenCV windows
cv2.destroyAllWindows()

# Stop any channels that are still playing
for channel in (channelOne, channelTwo, channelThree):
    if channel.get_busy():
        channel.stop()

# Shut down the Pygame mixer cleanly
pygame.mixer.quit()
