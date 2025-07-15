import logging
import librosa
import argparse
import os
import config

from MertEmbedder import MERTBeatEmbedder
from utils.beat_utils import compute_beat_similarity, compute_energy_per_beat
from utils.json_utils import JsonHandler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_beat_transitions(
        audio_path,
        audio_metadata: dict,
        beat_embedder: MERTBeatEmbedder,
        beat_similarity_threshold: float = 0.8,
        energy_similarity_threshold: float = 0.8,
        min_duration: float = 2.0,
        max_duration: float = 30.0,
):
    """
    Collects possible beat transitions from the audio file based on beat embeddings and energy levels.
    Args:
        audio_path: Path to the audio file (should be a .wav file).
        audio_metadata: Dictionary containing audio metadata, specifically the "beats" and "downbeats" keys.
        beat_embedder: Instance of MERTBeatEmbedder for computing beat embeddings.
        beat_similarity_threshold: threshold for beat embedding similarity to consider a transition valid.
        energy_similarity_threshold: threshold for energy similarity between beats to consider a transition valid.
        min_duration: Minimum duration for a beat transition in seconds.
        max_duration: Maximum duration for a beat transition in seconds.

    Returns:
        A dictionary containing the transitions between beats, their start times, and the similarity scores.
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    # Collect beat embeddings matrix
    beat_embeddings = beat_embedder(audio, sr, audio_metadata)
    beat_similarity_matrix = compute_beat_similarity(beat_embeddings)

    # Get beat info from metadata
    beat_times = audio_metadata.get("beats")
    down_beats = audio_metadata.get("downbeats")

    n_beats = len(beat_times)
    energy_tensor = compute_energy_per_beat(audio, sr, audio_metadata)
    transitions = []

    for i in range(n_beats):
        beat_transitions = []

        for j in range(i + 1, n_beats):
            if beat_times[i] not in down_beats or beat_times[j] not in down_beats: continue

            duration = beat_times[j] - beat_times[i]
            if duration < min_duration or duration > max_duration: continue

            emb_sim = beat_similarity_matrix[i, j].item()

            if emb_sim < beat_similarity_threshold: continue

            energy_i = energy_tensor[i].item()
            energy_j = energy_tensor[j].item()

            energy_sim = min(energy_i, energy_j) / max(energy_i, energy_j)

            if energy_sim < energy_similarity_threshold: continue

            beat_transitions.append({
                "to": j,
                "duration": round(duration, 2),
                "embedding_similarity": round(emb_sim, 2),
                "energy_similarity": round(energy_sim, 2)
            })

        if beat_transitions:
            transitions.append({
                "beat": i,
                "start_time": round(beat_times[i], 2),
                "transitions": beat_transitions
            })

    result = {
        "transitions": transitions,
        "beat_times": beat_times,
        "down_beats": down_beats
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Collect possible beat transitions from audio")

    parser.add_argument("audio_folder", type=str, help="Path to the folder with audio and metadata")

    args = parser.parse_args()

    audio_path = os.path.join(args.audio_folder, next((f for f in os.listdir(args.audio_folder) if f.endswith('.wav'))))
    metadata_path = os.path.join(args.audio_folder, "metadata.json")
    save_path = os.path.join(args.audio_folder, "transitions.json")

    beat_embedder = MERTBeatEmbedder(config.mert_model).to(config.device)

    beat_transitions = collect_beat_transitions(
        audio_path=audio_path,
        audio_metadata=JsonHandler.load(metadata_path),
        beat_embedder=beat_embedder,
        beat_similarity_threshold=config.beat_similarity_threshold,
        energy_similarity_threshold=config.energy_similarity_threshold,
        min_duration=config.min_duration,
        max_duration=config.max_duration,
    )

    JsonHandler.save(beat_transitions, file_path=save_path)
    logger.info(f"Transitions collected and saved to {save_path}")


if __name__ == "__main__":
    main()
