import cv2
from ultralytics import YOLO
import numpy as np
import torch
import pygame

# =============================================================================
# Audio Setup (concurrent sounds via Channels)
# =============================================================================

# Initialize the Pygame mixer for audio playback (must be done before loading sounds)
pygame.mixer.init()

# Create Sound objects for each song; keep names descriptive and in camelCase
songOneSound = pygame.mixer.Sound("song.mp3")    # Played when >= 1 person is detected
songTwoSound = pygame.mixer.Sound("song2.mp3")   # Played when >= 2 people are detected
songThreeSound = pygame.mixer.Sound("song3.mp3") # Played when >= 3 people are detected

# Set base volumes for each song (0.0 to 1.0)
songOneSound.set_volume(0.8)
songTwoSound.set_volume(0.8)
songThreeSound.set_volume(0.8)

# Reserve three channels so all three songs can play concurrently without interrupting each other
channelOne = pygame.mixer.Channel(0)
channelTwo = pygame.mixer.Channel(1)
channelThree = pygame.mixer.Channel(2)

# =============================================================================
# Detection Parameters
# =============================================================================

# Minimum confidence score required to consider a "PERSON" detection valid
personScoreThreshold = 0.40

# Optional stability filter: require a stable person count for N frames before changing audio
# Set to 0 for immediate reaction (no debounce)
stabilityFrameThreshold = 0  # Example: set to 3–5 to reduce rapid flicker of audio state

# Bookkeeping counters for stability filtering
stableFrameCount = 0             # Number of consecutive frames with the same person count
lastCommittedCount = 0           # Person count that has already been applied to audio
lastSeenCount = 0                # Most recent instantaneous person count seen

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
# Video Source (webcam 0 by default)
# =============================================================================

# Open the default camera; change the index if you have multiple cameras
videoCapture = cv2.VideoCapture(0)

# =============================================================================
# Audio Control Based on Person Count
# =============================================================================

def applyAudioForPersonCount(personCount: int) -> None:
    """
    Keep channel playback aligned with the current person count:
      >= 1 → channelOne loops songOneSound
      >= 2 → channelTwo loops songTwoSound
      >= 3 → channelThree loops songThreeSound
      < threshold → stop corresponding higher-index channels
    """
    # ---------------------------
    # Manage channelOne / songOne
    # ---------------------------
    if personCount >= 1:
        # If not already playing, start looping this track
        if not channelOne.get_busy():
            channelOne.play(songOneSound, loops=-1)
    else:
        # If currently playing but no longer needed, stop it
        if channelOne.get_busy():
            channelOne.stop()

    # ---------------------------
    # Manage channelTwo / songTwo
    # ---------------------------
    if personCount >= 2:
        if not channelTwo.get_busy():
            channelTwo.play(songTwoSound, loops=-1)
    else:
        if channelTwo.get_busy():
            channelTwo.stop()

    # -----------------------------
    # Manage channelThree / songThree
    # -----------------------------
    if personCount >= 3:
        if not channelThree.get_busy():
            channelThree.play(songThreeSound, loops=-1)
    else:
        if channelThree.get_busy():
            channelThree.stop()

# =============================================================================
# Main Loop: Read Frames, Run YOLO, Draw Results, Control Audio
# =============================================================================

while True:
    # Grab a frame from the camera
    didReadFrame, frameBgr = videoCapture.read()
    if not didReadFrame:
        # If frame read fails (camera unplugged, etc.), exit the loop gracefully
        break

    # Run YOLO inference on the raw frame
    # - device="mps" uses Apple Metal if available; adjust to "cuda" or "cpu" as needed
    inferenceResultsList = yoloModel(frameBgr, device="mps")
    inferenceResult = inferenceResultsList[0]  # We only need the first (and only) result

    # Extract detection tensors and convert to CPU numpy arrays for easy handling
    # - xyxy: bounding boxes in (left, top, right, bottom)
    # - cls: class indices
    # - conf: confidence scores
    boundingBoxes = np.array(inferenceResult.boxes.xyxy.cpu(), dtype="int")
    classIndices = np.array(inferenceResult.boxes.cls.cpu(), dtype="int")
    confidenceScores = np.array(inferenceResult.boxes.conf.cpu(), dtype="float")

    # Reset the per-frame count of valid PERSON detections
    currentPersonCount = 0

    # Iterate through all detections for this frame
    for boxCoordinates, classIndex, confidenceScore in zip(boundingBoxes, classIndices, confidenceScores):
        (left, top, right, bottom) = boxCoordinates  # Unpack box corners

        # Skip degenerate boxes that have invalid geometry
        if right - left <= 0 or bottom - top <= 0:
            continue

        # Draw a rectangle around the detection for visualization
        cv2.rectangle(frameBgr, (left, top), (right, bottom), (0, 0, 225), 2)

        # Prepare a human-readable label using the class name and confidence
        className = cocoClassMap.get(int(classIndex), "UNKNOWN").upper()
        labelText = f"{className}: {float(confidenceScore):.2f}"

        # Draw the label just above the bounding box
        cv2.putText(
            frameBgr,
            labelText,
            (left, top - 5),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 0, 225),
            2
        )

        # Increase the count if this detection is a sufficiently confident PERSON
        if className == "PERSON" and float(confidenceScore) >= personScoreThreshold:
            currentPersonCount += 1

    # -----------------------------------------------------------------------------
    # Optional stability debounce: require a stable count for N frames before commit
    # -----------------------------------------------------------------------------
    if stabilityFrameThreshold <= 0:
        # Immediate mode: apply the audio state every frame based on the current count
        applyAudioForPersonCount(currentPersonCount)
        lastCommittedCount = currentPersonCount
    else:
        # Debounced mode: only change audio after N consecutive frames with same count
        if currentPersonCount == lastSeenCount:
            stableFrameCount += 1  # Count another same-value frame
        else:
            # Count changed: reset stability counter and track the new value
            stableFrameCount = 0
            lastSeenCount = currentPersonCount

        # When the same count persists for enough frames, apply it
        if stableFrameCount >= stabilityFrameThreshold and currentPersonCount != lastCommittedCount:
            applyAudioForPersonCount(currentPersonCount)
            lastCommittedCount = currentPersonCount
            stableFrameCount = 0  # Reset so we can detect future changes cleanly

    # -----------------------------------------------------------------------------
    # On-screen status overlay: show the computed person count in the top-left
    # -----------------------------------------------------------------------------
    cv2.putText(
        frameBgr,
        f"PERSON count: {currentPersonCount}",
        (20, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display the annotated frame in a window
    cv2.imshow("YOLO Detection", frameBgr)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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
