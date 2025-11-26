#!/usr/bin/env python3
"""
Push the MP3 file name over serial so the Feather ESP32-S2 TFT can display it.
Also listens for play commands and plays MP3 files on the laptop.
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import serial

AUDIO_EXTENSIONS = (".mp3", ".wav", ".aiff", ".aif", ".flac", ".aac", ".m4a", ".ogg")
TRACK_SEPARATOR = "\t"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "port",
        help="Serial device path, e.g. /dev/cu.usbmodem14101",
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Explicit message to send (skips directory scanning when provided)",
    )
    parser.add_argument(
        "-b",
        "--baud",
        type=int,
        default=115200,
        help="Baud rate (default: %(default)s)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait after opening the port before sending data",
    )
    parser.add_argument(
        "-d",
        "--dir",
        default="Music",
        help="Directory to scan for audio files when no message argument is passed "
        "(relative paths resolve against this script)",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=None,
        help="Zero-based index into the alphabetized list of discovered files "
        "(omit to send the full list)",
    )
    parser.add_argument(
        "--strip-extension",
        action="store_true",
        help="Send only the filename stem (omit file extension)",
    )
    return parser.parse_args()


def resolve_directory(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def find_audio_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    candidates = sorted(
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not candidates:
        raise FileNotFoundError(
            f"No audio files ({', '.join(AUDIO_EXTENSIONS)}) found in {directory}"
        )
    return candidates


def gather_tracks(args: argparse.Namespace) -> Tuple[List[str], List[Path]]:
    """Returns (track_names, audio_file_paths) tuple."""
    if args.message:
        candidate = args.message.strip()
        # For explicit messages, we can't determine the file path
        return ([candidate] if candidate else []), []

    directory = resolve_directory(args.dir)
    files = find_audio_files(directory)
    names = [
        file_path.stem if args.strip_extension else file_path.name for file_path in files
    ]

    if args.index is not None:
        try:
            return [names[args.index]], [files[args.index]]
        except IndexError as exc:
            raise IndexError(
                f"Index {args.index} is out of range for {len(names)} discovered files"
            ) from exc

    return names, files


def encode_payload(messages: List[str]) -> bytes:
    filtered = [entry.strip() for entry in messages if entry.strip()]
    if not filtered:
        raise ValueError("No valid track names to send")
    joined = TRACK_SEPARATOR.join(filtered)
    return (joined + "\n").encode("utf-8")


# Global variables to track playing processes
playback_processes: Dict[int, subprocess.Popen] = {}
playback_lock = threading.Lock()
serial_port: Optional[serial.Serial] = None


def send_status_to_arduino(message: str) -> None:
    """Send a status message to the Arduino."""
    global serial_port
    if serial_port is not None:
        try:
            serial_port.write((message + "\n").encode("utf-8"))
            serial_port.flush()
        except Exception as exc:
            print(f"Error sending status to Arduino: {exc}", file=sys.stderr)


def monitor_playback(index: int, process: subprocess.Popen) -> None:
    """Monitor a playback process and send status updates when it completes."""
    try:
        process.wait()  # Wait for the process to complete
        with playback_lock:
            if index in playback_processes and playback_processes[index] == process:
                del playback_processes[index]
                send_status_to_arduino(f"STOPPED:{index}")
                print(f"Track {index} finished playing")
    except Exception as exc:
        print(f"Error monitoring playback for track {index}: {exc}", file=sys.stderr)


def play_mp3(file_path: Path, index: int) -> None:
    """Play an MP3 file using afplay (macOS built-in player)."""
    global playback_processes
    
    try:
        process = subprocess.Popen(
            ["afplay", str(file_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        with playback_lock:
            playback_processes[index] = process
        
        # Send status to Arduino
        send_status_to_arduino(f"PLAYING:{index}")
        print(f"Playing: {file_path.name} (index {index})")
        
        # Start a thread to monitor when playback completes
        monitor_thread = threading.Thread(
            target=monitor_playback, args=(index, process), daemon=True
        )
        monitor_thread.start()
        
    except Exception as exc:
        print(f"Error playing {file_path.name}: {exc}", file=sys.stderr)


def pause_playback(index: int) -> None:
    """Pause a specific track by terminating its playback process."""
    global playback_processes
    
    with playback_lock:
        if index in playback_processes:
            process = playback_processes[index]
            try:
                process.terminate()
                process.wait(timeout=1.0)
                print(f"Track {index} paused")
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                print(f"Track {index} force-stopped")
            except Exception as exc:
                print(f"Error pausing track {index}: {exc}", file=sys.stderr)
            finally:
                del playback_processes[index]
                send_status_to_arduino(f"STOPPED:{index}")


def handle_play_command(
    command: str, audio_files: List[Path], directory: Path
) -> None:
    """Handle a PLAY command from the Arduino."""
    try:
        # Command format: "PLAY:index"
        index = int(command.split(":")[1])
        if 0 <= index < len(audio_files):
            file_path = audio_files[index]
            play_mp3(file_path, index)
        else:
            print(f"Invalid track index: {index}", file=sys.stderr)
    except (ValueError, IndexError) as exc:
        print(f"Invalid play command format: {command} - {exc}", file=sys.stderr)


def handle_pause_command(command: str) -> None:
    """Handle a PAUSE command from the Arduino."""
    try:
        # Command format: "PAUSE:index"
        index = int(command.split(":")[1])
        pause_playback(index)
    except (ValueError, IndexError) as exc:
        print(f"Invalid pause command format: {command} - {exc}", file=sys.stderr)


def listen_for_commands(
    ser: serial.Serial, audio_files: List[Path], directory: Path
) -> None:
    """Continuously listen for commands from the Arduino."""
    global serial_port
    serial_port = ser  # Store reference for status updates
    
    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("PLAY:"):
                    handle_play_command(line, audio_files, directory)
                elif line.startswith("PAUSE:"):
                    handle_pause_command(line)
            time.sleep(0.1)  # Small delay to prevent CPU spinning
        except serial.SerialException:
            break  # Serial port closed
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"Error reading serial: {exc}", file=sys.stderr)
            time.sleep(0.5)


def main() -> int:
    args = parse_args()
    
    try:
        tracks, audio_files = gather_tracks(args)
        payload = encode_payload(tracks)
    except (OSError, IndexError, ValueError) as exc:
        print(f"Cannot determine filenames to send: {exc}", file=sys.stderr)
        return 2

    # If no audio files were found (e.g., when using --message), 
    # try to get them from the directory anyway for playing
    if not audio_files:
        directory = resolve_directory(args.dir)
        try:
            audio_files = find_audio_files(directory)
        except (FileNotFoundError, NotADirectoryError):
            pass  # Will handle gracefully in listen_for_commands

    try:
        with serial.Serial(args.port, args.baud, timeout=2) as ser:
            time.sleep(args.delay)  # allow the Feather to reboot after USB open
            ser.write(payload)
            ser.flush()
            
            if len(tracks) == 1:
                print(f"Sent {tracks[0]!r} to {args.port}")
            else:
                print(f"Sent {len(tracks)} tracks to {args.port}")
            
            if audio_files:
                print("Listening for play commands... (Press Ctrl+C to exit)")
                # Start listening for commands
                listen_for_commands(ser, audio_files, resolve_directory(args.dir))
            else:
                print("No audio files found - cannot play tracks")
            
    except serial.SerialException as exc:
        print(f"Failed to open {args.port}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

