import argparse
import os
from pydub import AudioSegment


def load_and_prepare(path, target_frame_rate=None,
                     target_channels=None, target_sample_width=None):
    """
    Load an audio file and (optionally) convert it to match
    the target frame rate, channels, and sample width.
    """
    audio = AudioSegment.from_file(path)

    if target_frame_rate is not None:
        audio = audio.set_frame_rate(target_frame_rate)
    if target_channels is not None:
        audio = audio.set_channels(target_channels)
    if target_sample_width is not None:
        audio = audio.set_sample_width(target_sample_width)

    return audio


def mix_two_songs(
    song1_path,
    song2_path,
    out_path,
    gain1_db=0.0,
    gain2_db=0.0,
    crossfade_ms=0
):
    """
    Mix so that song1 transitions into song2.

    - If crossfade_ms == 0:
        [song1][song2]
    - If crossfade_ms > 0:
        [song1_without_tail][crossfaded_overlap][rest_of_song2]
    """

    # Load first song (defines target parameters)
    song1 = AudioSegment.from_file(song1_path)

    # Load second song and match parameters to song1
    song2 = load_and_prepare(
        song2_path,
        target_frame_rate=song1.frame_rate,
        target_channels=song1.channels,
        target_sample_width=song1.sample_width,
    )

    # Apply volume adjustments (in dB)
    song1 = song1 + gain1_db
    song2 = song2 + gain2_db

    # Ensure crossfade duration is not longer than either track
    if crossfade_ms > 0:
        max_possible = min(len(song1), len(song2))
        if crossfade_ms > max_possible:
            crossfade_ms = max_possible

    # ---- Build final mix ----
    if crossfade_ms > 0:
        # Split song1: everything except its last crossfade_ms
        song1_main = song1[:-crossfade_ms]
        song1_tail = song1[-crossfade_ms:]

        # Split song2: first crossfade_ms and the remainder
        song2_intro = song2[:crossfade_ms]
        song2_rest = song2[crossfade_ms:]

        # Fade out end of song1, fade in beginning of song2
        song1_tail_faded = song1_tail.fade_out(crossfade_ms)
        song2_intro_faded = song2_intro.fade_in(crossfade_ms)

        # Overlap the faded tail and intro
        crossfaded_section = song1_tail_faded.overlay(song2_intro_faded)

        # Final structure: main of song1 + crossfaded overlap + rest of song2
        mixed = song1_main + crossfaded_section + song2_rest

    else:
        # Simple hard cut: song1 then song2
        mixed = song1 + song2

    # Export
    file_root, ext = os.path.splitext(out_path)
    if ext == "":
        # default to mp3
        out_path = file_root + ".mp3"

    mixed.export(out_path, format=out_path.split(".")[-1])
    print(f"✅ Mixed track saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Mix two songs so the first transitions into the second."
    )
    parser.add_argument("song1", help="Path to first song (e.g., song1.mp3)")
    parser.add_argument("song2", help="Path to second song (e.g., song2.mp3)")
    parser.add_argument(
        "-o", "--out",
        default="mixed_output.mp3",
        help="Output file path (default: mixed_output.mp3)"
    )
    parser.add_argument(
        "--gain1",
        type=float,
        default=0.0,
        help="Gain (in dB) to apply to song1 (e.g., -6.0 to make quieter)"
    )
    parser.add_argument(
        "--gain2",
        type=float,
        default=0.0,
        help="Gain (in dB) to apply to song2"
    )
    parser.add_argument(
        "--crossfade",
        type=int,
        default=0,
        help="Crossfade duration in ms (0 = no crossfade)"
    )

    args = parser.parse_args()

    mix_two_songs(
        song1_path=args.song1,
        song2_path=args.song2,
        out_path=args.out,
        gain1_db=args.gain1,
        gain2_db=args.gain2,
        crossfade_ms=args.crossfade,
    )


if __name__ == "__main__":
    main()
