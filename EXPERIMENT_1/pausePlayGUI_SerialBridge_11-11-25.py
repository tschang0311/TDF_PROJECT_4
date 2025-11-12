#   TECHNOLOGY DESING FOUNDATIONS - PROJECT 4
#   THOMAS CHANG, KENYA FOSTER, BRYCE PARSONS
#   11/11/25

#   ESP32 I/O Device to Python Controller

#   The following program has been cleaned and commented
#   for readibility with the help of ChatGPT. Additionally,
#   this program was vibe-coded with the help of ChatGPT.
#   AI assistance can help us rapidly experiment and test
#   our ideas.

import threading
import json
import re
import time
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import pygame
import serial


class Esp32DjControllerApp:
    """
    ESP32 DJ Controller (Tkinter + Pygame + PySerial)

    - Tkinter: simple desktop UI for loading and controlling audio
    - Pygame mixer: audio playback engine
    - PySerial: listens for ESP32 I/O events in the form: IO:<DEVICE>:<ACTION>
        Example events:
            IO:BTN_A:PRESS
            IO:BTN_A:LONG
            IO:ENC1:CW
    """

    def __init__(self, rootWindow: tk.Tk):
        # ---------- Audio ----------
        pygame.mixer.init()  # Initialize the global Pygame mixer (single process-wide audio engine)

        # ---------- UI ----------
        self.rootWindow = rootWindow                     # Keep a reference to the main window
        self.rootWindow.title("ESP32 DJ")                # Window title
        self.audioFilePath = None                        # Full path to the current audio file
        self.isPlaying = False                           # True if audio is currently playing (or unpaused)

        # Buttons for basic controls
        tk.Button(rootWindow, text="Select Song", command=self.loadSong).pack(fill="x")
        tk.Button(rootWindow, text="Play",         command=self.play).pack(fill="x")
        tk.Button(rootWindow, text="Pause",        command=self.pause).pack(fill="x")
        tk.Button(rootWindow, text="Connect ESP32", command=self.connectSerial).pack(fill="x")

        # Status label to show file path or serial status
        self.statusLabel = tk.Label(rootWindow, anchor="w")
        self.statusLabel.pack(fill="x")

        # Intercept the window close button to clean up resources
        rootWindow.protocol("WM_DELETE_WINDOW", self.close)

        # ---------- Serial ----------
        self.serialPort = None  # Will hold an instance of serial.Serial upon connection

        # ---------- I/O Behavior Registry ----------
        # Map of (DEVICE, ACTION) → callback; keys stored in uppercase
        self.ioBehaviors = {}
        # Fallback callback if a specific mapping is not found; signature: callback(device, action)
        self.defaultIoBehavior = None

        # Default mappings for a simple single button:
        self.registerBehavior("BTN_A", "PRESS", self.togglePlayPause)  # Short press toggles play/pause
        self.registerBehavior("BTN_A", "LONG",  self.pause)            # Long press pauses

    # ==================================================================================
    # UI / Audio
    # ==================================================================================
    def loadSong(self):
        """
        Prompt the user to pick an audio file and load it into the mixer.
        """
        # Restrict to common audio formats supported by Pygame mixer
        chosenPath = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.ogg")])
        if chosenPath:
            self.audioFilePath = chosenPath  # Save the selected path for later Play/Unpause
            pygame.mixer.music.load(chosenPath)  # Load into Pygame's music stream
            self.statusLabel.config(text=chosenPath)  # Update UI with path
            print(f"[AUDIO] Loaded: {chosenPath}")

    def play(self):
        """
        Begin playback (or unpause if already loaded and paused).
        """
        # If no file has been selected yet, prompt the user
        if not self.audioFilePath:
            self.loadSong()
        # If the user still did not choose a file, bail out
        if not self.audioFilePath:
            return

        # If something was already playing and is now paused, unpause
        if self.isPlaying:
            pygame.mixer.music.unpause()
            print("[AUDIO] Unpause")
        else:
            # Start fresh playback from the beginning of the loaded file
            pygame.mixer.music.play()
            print("[AUDIO] Play")

        # Mark state so UI and handlers know we are in a playing state
        self.isPlaying = True

    def pause(self):
        """
        Pause playback if the music channel is active.
        """
        # Only pause if music is currently active; get_busy() is True while the mixer is engaged
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            print("[AUDIO] Pause")
        # Update state to reflect not-playing (paused) condition
        self.isPlaying = False

    def stop(self):
        """
        Stop playback entirely and reset isPlaying state.
        """
        pygame.mixer.music.stop()
        print("[AUDIO] Stop")
        self.isPlaying = False

    def togglePlayPause(self, *maybeEventArgs):
        """
        Toggle between play and pause.
        Accepts optional args so it can be used as a generic I/O callback.
        """
        print("[AUDIO] Toggle play/pause")
        # If currently playing, pause; otherwise, attempt to play (or unpause)
        self.pause() if self.isPlaying else self.play()

    def setVolumeDelta(self, deltaPercent: float):
        """
        Adjust volume by a relative percentage (-100..+100 mapped to -1.0..+1.0 in mixer).
        Clamps the final volume to 0.0..1.0.
        """
        # Get current volume (float 0.0..1.0)
        currentVolume = pygame.mixer.music.get_volume()
        # Convert percentage delta to the 0..1 range step
        newVolume = currentVolume + (deltaPercent / 100.0)
        # Clamp to valid bounds
        newVolume = max(0.0, min(1.0, newVolume))
        # Apply to the mixer
        pygame.mixer.music.set_volume(newVolume)
        print(f"[AUDIO] Volume: {int(newVolume * 100)}%")

    # ==================================================================================
    # Serial / Connection
    # ==================================================================================
    def connectSerial(self):
        """
        Ask the user for a serial port, open it, optionally push a device config,
        and start the background thread that reads serial lines.
        """
        # If already connected, do not reconnect
        if self.serialPort and self.serialPort.is_open:
            print("[INFO] Serial already connected.")
            return

        # Prompt the user for a port (platform-dependent names)
        portName = simpledialog.askstring(
            "ESP32 Port",
            "Enter port (e.g. COM3 or /dev/ttyUSB0 or /dev/tty.usbserial-XXXX):"
        )
        if not portName:
            return  # User canceled

        try:
            # Open the serial port at 115200 baud with a 1-second read timeout
            self.serialPort = serial.Serial(portName, 115200, timeout=1)
            print(f"[INFO] Serial port connected successfully: {portName}")
            time.sleep(0.2)                    # Small settle delay to allow boot noise to finish
            self.serialPort.reset_input_buffer()  # Clear any garbage from the input buffer
        except serial.SerialException as error:
            # Show an error dialog and log to console; keep serialPort as None
            messagebox.showerror("Serial Error", str(error))
            print(f"[ERROR] Failed to connect to serial port: {error}")
            self.serialPort = None
            return

        # Update the status label to indicate a healthy connection
        self.statusLabel.config(text=f"Serial connected: {portName}")

        # Optionally send a device configuration to the ESP32 Arduino sketch (if it supports ArduinoJson)
        self.sendDeviceConfig({
            "cmd": "cfg",
            "device": "BTN_A",
            "pin": 12,
            "mode": "INPUT_PULLUP",
            "debounce_ms": 60,
            "long_ms": 500
        })

        # Start a daemon thread to continuously read lines from the serial port
        threading.Thread(target=self.readSerial, daemon=True).start()

    def _sanitizeCommand(self, rawString: str) -> str:
        """
        Normalize a line from serial:
          - strip surrounding whitespace
          - remove non-printable characters
          - collapse internal whitespace
        Return an empty string if nothing meaningful remains.
        """
        # If None or empty, return empty string to signal "ignore"
        if not rawString:
            return ""
        # Remove leading/trailing whitespace
        sanitized = rawString.strip()
        # Drop non-printable characters across the string
        sanitized = "".join(ch for ch in sanitized if ch.isprintable())
        # Squash stretches of whitespace to one space
        sanitized = re.sub(r"\s+", " ", sanitized)
        return sanitized

    def readSerial(self):
        """
        Background thread:
          - Continuously read lines from the serial port
          - Sanitize and then forward each command to the main thread via Tk's .after()
        """
        while self.serialPort and self.serialPort.is_open:
            try:
                # Read one line (until \n) or timeout (returns b"")
                rawBytes = self.serialPort.readline()
            except serial.SerialException:
                # If the serial port errors or closes, break out of the loop
                break

            # If nothing was read (timeout), continue waiting
            if not rawBytes:
                continue

            try:
                # Decode bytes to text using UTF-8; ignore invalid sequences
                decodedText = rawBytes.decode("utf-8", errors="ignore")
            except Exception:
                # If decoding fails for any reason, skip this line
                continue

            # Clean up the text to a predictable, minimal form
            sanitizedCommand = self._sanitizeCommand(decodedText)
            if not sanitizedCommand:
                continue  # Ignore empty or meaningless content

            # Log the raw command for debugging
            print(f"[CMD] Received from ESP32: {sanitizedCommand}")

            # Forward the command to the GUI thread for handling (never touch UI from background threads)
            self.rootWindow.after(0, lambda c=sanitizedCommand: self.handleSerial(c))

        # If we exit the loop, the serial connection is gone; notify the UI on the main thread
        print("[INFO] Serial port disconnected.")
        self.rootWindow.after(0, lambda: self.statusLabel.config(text="Serial disconnected"))

    def sendDeviceConfig(self, configDictionary: dict):
        """
        Send a JSON configuration object to the ESP32, if the serial connection is open.
        """
        # Make sure the port is open before attempting to write
        if not (self.serialPort and self.serialPort.is_open):
            print("[WARN] Cannot send config; serial not connected.")
            return

        try:
            # Serialize dictionary to a JSON line and add newline to delimit the message
            outgoingLine = json.dumps(configDictionary) + "\n"
            # Encode as UTF-8 and write to the port
            self.serialPort.write(outgoingLine.encode("utf-8"))
            print(f"[INFO] Sent config → {outgoingLine.strip()}")
        except serial.SerialException as error:
            print(f"[ERROR] Failed to send config: {error}")

    # ==================================================================================
    # Serial Command Handling
    # ==================================================================================
    def handleSerial(self, commandText: str):
        """
        Handle a sanitized line from the serial device.

        Supported formats:
          1) I/O events (preferred):  IO:<DEVICE>:<ACTION>
             - Routes to self.handleIoEvent()
          2) Legacy commands: PLAY | PAUSE | TOGGLE
             - Directly mapped to audio control methods
        """
        # Re-sanitize defensively in case this method is called from other places
        cleanedCommand = self._sanitizeCommand(commandText)
        if not cleanedCommand:
            return  # Nothing to do

        # I/O events of the form "IO:DEVICE:ACTION"
        if cleanedCommand.startswith("IO:"):
            parts = cleanedCommand.split(":")
            # Require at least IO, DEVICE, ACTION
            if len(parts) >= 3:
                deviceName, actionName = parts[1], parts[2]
                self.handleIoEvent(deviceName, actionName)
                return  # Done

        # Legacy simple commands (case-insensitive)
        upperCommand = cleanedCommand.upper()
        if upperCommand == "PLAY":
            self.play()
        elif upperCommand == "PAUSE":
            self.pause()
        elif upperCommand == "TOGGLE":
            self.togglePlayPause()
        else:
            print(f"[WARN] Unknown command: {cleanedCommand}")

    # ==================================================================================
    # I/O Behavior Registry
    # ==================================================================================
    def registerBehavior(self, deviceName: str, actionName: str, callback):
        """
        Register a behavior for a given (device, action) pair.
        Example:
            registerBehavior("BTN_A", "PRESS", self.togglePlayPause)
        """
        # Normalize keys to uppercase to keep lookups case-insensitive
        behaviorKey = (deviceName.upper(), actionName.upper())
        self.ioBehaviors[behaviorKey] = callback

        # Log the binding for visibility; try to print the callback's name when possible
        print(f"[BIND] {behaviorKey} → {getattr(callback, '__name__', str(callback))}")

    def unregisterBehavior(self, deviceName: str, actionName: str):
        """
        Remove a previously registered behavior for (device, action), if present.
        """
        self.ioBehaviors.pop((deviceName.upper(), actionName.upper()), None)

    def setDefaultBehavior(self, callback):
        """
        Set a fallback behavior used when no mapping exists for a received (device, action).
        Expected signature: callback(device, action)
        """
        self.defaultIoBehavior = callback

    def handleIoEvent(self, deviceName: str, actionName: str):
        """
        Dispatch an I/O event to a registered behavior.
        Tries exact (device, action) match; otherwise, uses the default behavior if available.
        """
        # Normalize for registry key lookup
        behaviorKey = (deviceName.upper(), actionName.upper())
        callback = self.ioBehaviors.get(behaviorKey)

        if callback is not None:
            # Log the dispatch and invoke the callback
            print(f"[IO] {deviceName}:{actionName} → {getattr(callback, '__name__', str(callback))}")
            try:
                # Prefer callbacks that accept (device, action) for richer context
                callback(deviceName, actionName)
            except TypeError:
                # Fall back to zero-argument callbacks if necessary
                callback()
        elif self.defaultIoBehavior:
            # Use the fallback when no specific mapping exists
            print(f"[IO] {deviceName}:{actionName} → default")
            self.defaultIoBehavior(deviceName, actionName)
        else:
            # Nothing mapped; warn for visibility
            print(f"[IO] No behavior mapped for {deviceName}:{actionName}")

    # ==================================================================================
    # Cleanup
    # ==================================================================================
    def close(self):
        """
        Gracefully close the serial port (if open), stop audio, and destroy the Tkinter window.
        """
        # Attempt to close serial cleanly
        if self.serialPort and self.serialPort.is_open:
            print("[INFO] Closing serial port...")
            try:
                self.serialPort.close()
            except Exception:
                # Ignore close errors; port might already be gone
                pass

        # Stop audio and reset state
        self.stop()

        # Destroy the Tkinter window and end the application
        print("[INFO] Application closed.")
        self.rootWindow.destroy()


if __name__ == "__main__":
    # Create and run the app in the Tkinter mainloop
    Esp32DjControllerApp(tk.Tk()).rootWindow.mainloop()
