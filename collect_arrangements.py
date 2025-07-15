import argparse
import config

from utils.json_utils import JsonHandler
from ArrangementSystem import ArrangementSystem
from utils.arrangement_utils import save_arrangements_to_json, print_arrangement


def main():
    parser = argparse.ArgumentParser(description="Collect arrangements based on configuration.")

    parser.add_argument(
        "audio_folder",
        type=str,
        help="Path to the audio folder containing audio file and metadata."
    )

    parser.add_argument(
        "--target_duration",
        type=float,
        required=True,
        help="Target duration for the arrangements."
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Enable verbose output."
    )

    args = parser.parse_args()

    metadata_path = f"{args.audio_folder}/metadata.json"
    metadata = JsonHandler.load(metadata_path)

    transitions_path = f"{args.audio_folder}/transitions.json"
    transitions = JsonHandler.load(transitions_path)

    arrangement_system = ArrangementSystem(config)

    arrangements = arrangement_system.find_arrangements(
        args.target_duration,
        transitions,
        metadata
    )

    save_arrangements_to_json(arrangements, args.audio_folder + "/arrangements.json")

    if args.verbose:
        for idx, arrangement in enumerate(arrangements, start=1):
            print_arrangement(arrangement, index=idx)


if __name__ == "__main__":
    main()
