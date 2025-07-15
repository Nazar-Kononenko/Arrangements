import os

import allin1
import soundfile as sf
import argparse

from utils.json_utils import JsonHandler


def main():
    parser = argparse.ArgumentParser(description="Analyze audio files using allin1.")

    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to analyze."
    )

    args = parser.parse_args()

    audio, sample_rate = sf.read(args.audio_file)
    duration = len(audio) / sample_rate

    result = allin1.analyze(args.audio_file)

    metadata = {
        "duration": duration,
        "bpm": result.bpm,
        "path": str(result.path),
        "beats": result.beats,
        "downbeats": result.downbeats,
        'segments': [
            {'start': float(seg.start), 'end': float(seg.end), 'label': seg.label}
            for seg in result.segments
        ]}

    file_name = os.path.splitext(os.path.basename(args.audio_file))[0]

    os.makedirs(file_name, exist_ok=True)

    new_audio_path = os.path.join(file_name, os.path.basename(args.audio_file))
    sf.write(new_audio_path, audio, sample_rate)

    JsonHandler.save(metadata, f"{file_name}/metadata.json")


if __name__ == "__main__":
    main()
