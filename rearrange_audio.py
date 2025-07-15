import argparse
import os

from utils.arrangement_utils import rearrange_audio, ArrangementPath
from utils.json_utils import JsonHandler


def main():
    parser = argparse.ArgumentParser(description="Rearrange audio file")

    parser.add_argument(
        "audio_folder",
        type=str,
        help="Path to the folder containing the audio file and metadata"
    )

    parser.add_argument(
        "arrangement_index",
        type=str,
        help="Index of the arrangement to apply (1-based)"
    )

    args = parser.parse_args()

    audio_path = os.path.join(args.audio_folder, next((f for f in os.listdir(args.audio_folder) if f.endswith('.wav'))))
    metadata_path = os.path.join(args.audio_folder, "metadata.json")
    arrangements_path = os.path.join(args.audio_folder, "arrangements.json")
    save_path = os.path.join(args.audio_folder, f"arrangement_{args.arrangement_index}.wav")

    metadata = JsonHandler.load(metadata_path)
    arrangements = JsonHandler.load(arrangements_path)

    arrangement = next(
        (arr for arr in arrangements if arr["index"] == int(args.arrangement_index)),
        None
    )

    if arrangement is None:
        raise ValueError(f"No arrangement found with index {args.arrangement_index}")

    arrangement_path = ArrangementPath.from_dict(arrangement)

    rearrange_audio(
        audio_path=audio_path,
        audio_metadata=metadata,
        arrangement=arrangement_path,
        output_path=save_path
    )


if __name__ == "__main__":
    main()
