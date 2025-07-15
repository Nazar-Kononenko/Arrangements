import logging

from dataclasses import dataclass
from typing import List, Dict, Any
from utils.json_utils import JsonHandler
from pydub import AudioSegment


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ArrangementLengths:
    original_duration: float
    adjusted_duration: float
    target_duration: float


@dataclass
class ArrangementPath:
    transitions: List[Dict[str, Any]]
    lengths: ArrangementLengths
    score: float

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        return ArrangementPath(
            transitions=data.get("transitions"),
            lengths=ArrangementLengths(
                original_duration=data.get("original_duration"),
                adjusted_duration=data.get("arrangement_duration"),
                target_duration=data.get("target_duration")
            ),
            score=data.get("score")
        )


def has_redundant_chain(path: ArrangementPath, beat_graph: Dict[int, dict]) -> bool:
    """
    Check if the arrangement path contains a redundant chain of skip transitions.

    Args:
        path: ArrangementPath, the arrangement path to check.
        beat_graph: Dict[int, dict], the graph of beats and their transitions.

    Returns:
        bool: True if a redundant chain is found, False otherwise.
    """
    for i in range(len(path.transitions) - 1):
        a = path.transitions[i]
        b = path.transitions[i + 1]

        if a['type'] == 'skip' and b['type'] == 'skip' and a['to_beat'] == b['from_beat']:
            start = a['from_beat']
            end = b['to_beat']

            for t in beat_graph.get(start, []):
                if t['to'] == end:
                    return True

    return False


def save_arrangements_to_json(arrangements: List[ArrangementPath], filename: str):
    """
    Save the arrangement paths to a JSON file.

    Args:
        arrangements: List[ArrangementPath], the list of arrangement paths to save.
        filename: str, the name of the file to save the arrangements to.
    """
    json_data = [
        {
            "index": idx,
            "transitions": path.transitions,
            "arrangement_duration": path.lengths.adjusted_duration,
            "original_duration": path.lengths.original_duration,
            "target_duration": path.lengths.target_duration,
            "score": path.score
        } for idx, path in enumerate(arrangements, start=1)
    ]

    JsonHandler.save(json_data, filename)
    logger.info(f"Arrangements saved to {filename}")


def print_arrangement(path: ArrangementPath, index=1):
    """
    Print the arrangement path in a formatted way.

    Args:
        path: ArrangementPath, the arrangement path to print.
        index: int, the index of the arrangement path for display purposes.
    """
    headers = ["Type", "From Beat", "To Beat", "Duration"]
    col_widths = [7, 10, 10, 10]

    header_row = " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
    separator = "-+-".join("-" * width for width in col_widths)

    transition_rows = "\n".join(
        f"{action['type']:<7} | {action['from_beat']:<10} | {action['to_beat']:<10} | {action['duration']:<10.2f}"
        for action in path.transitions
    )

    print(f"{'=' * 50}\n"
          f"Arrangement {index}\n"
          f"{'-' * 50}\n"
          f"Original Audio Length: {path.lengths.original_duration:.2f} seconds\n"
          f"Target Length: {path.lengths.target_duration:.2f} seconds\n"
          f"Arrangement Length: {path.lengths.adjusted_duration:.2f} seconds\n"
          f"Score: {path.score:.3f}\n"
          f"{'-' * 50}\n"
          f"Transitions:\n"
          f"{header_row}\n"
          f"{separator}\n"
          f"{transition_rows}\n"
          f"{'-' * 50}\n"
          f"Result: {path.lengths.adjusted_duration:.2f} seconds\n"
          f"{'=' * 50}")

    print()


def rearrange_audio(audio_path: str, audio_metadata: dict, arrangement: ArrangementPath, output_path: str,
                    audio_format="wav"):
    """
    Rearrange the audio file according to the specified arrangement.

    Args:
        audio_path: str, path to the original audio file.
        audio_metadata: dict, metadata containing beats and segments information.
        arrangement: ArrangementPath, the arrangement path containing transitions.
        output_path: str, path to save the rearranged audio file.
        audio_format: str, format of the audio file to save (default is "wav").

    Returns:
        AudioSegment: The rearranged audio segment after applying the all transitions and loops from the arrangement.
    """
    beats = audio_metadata.get("beats")
    original_duration = audio_metadata.get("duration")
    pydub_audio = AudioSegment.from_file(audio_path, format=audio_format)

    used_beats = set()
    segments = []

    for action in arrangement.transitions:
        used_beats.add(action["from_beat"])
        used_beats.add(action["to_beat"])

    first_beat = min(used_beats) if used_beats else 0
    last_beat = max(used_beats) if used_beats else len(beats) - 1

    if first_beat > 0 or not arrangement.transitions:
        start_time = 0
        end_time = int(beats[first_beat] * 1000) if used_beats else int(original_duration * 1000)
        segments.append(pydub_audio[start_time:end_time])

    current_beat = first_beat

    for action in arrangement.transitions:
        from_beat = action["from_beat"]
        to_beat = action["to_beat"]

        if current_beat < from_beat:
            start_time = int(beats[current_beat] * 1000)
            end_time = int(beats[from_beat] * 1000)
            segments.append(pydub_audio[start_time:end_time])

        if action["type"] == "skip":
            current_beat = to_beat
        else:
            start_time = int(beats[from_beat] * 1000)
            end_time = int(beats[to_beat] * 1000)
            segment = pydub_audio[start_time:end_time]

            for _ in range(action["loop_count"]):
                segments.append(segment)

            current_beat = to_beat

    if current_beat < len(beats) and beats[last_beat] < original_duration:
        start_time = int(beats[last_beat] * 1000)
        end_time = int(original_duration * 1000)
        segments.append(pydub_audio[start_time:end_time])

    final_audio = sum(segments, AudioSegment.empty())

    final_audio.export(output_path, format="wav")

    logger.info(f"Rearranged audio and saved to {output_path}")

    return final_audio
