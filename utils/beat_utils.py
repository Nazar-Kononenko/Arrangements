import torch
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from typing import Dict, Any, Optional, List


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_beat_similarity(beat_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity matrix for the given beat embeddings.
    Args:
        beat_embeddings: A 2D tensor of shape (num_beats, embedding_dim) containing the embeddings for each beat.
    Returns:
        A 2D tensor representing the cosine similarity matrix of shape (num_beats, num_beats).
    """
    embeddings_norm = F.normalize(beat_embeddings, p=2, dim=1)
    similarity_matrix = embeddings_norm @ embeddings_norm.T

    logger.info(f"Computed cosine similarity matrix of shape {similarity_matrix.shape}")

    return similarity_matrix


def compute_energy_per_beat(audio: np.ndarray, sr: int, audio_metadata: dict) -> torch.Tensor:
    """
    Computes the energy of each beat in the audio based on the provided beat timestamps.
    Args:
        audio: numpy array of audio samples
        sr: sampling rate of the audio
        audio_metadata: dictionary containing audio metadata, specifically the "beats" key which should be a list of beat timestamps.

    Returns:
        A tensor containing the energy for each beat.
    """
    beat_times = audio_metadata.get("beats")

    if not isinstance(beat_times, list) or len(beat_times) < 2:
        raise ValueError("beat_times should be a list with at least two elements")

    energy_per_beat = []
    n_frames = len(audio)

    for i in range(len(beat_times)):
        start_time = beat_times[i]
        end_time = beat_times[i + 1] if i + 1 < len(beat_times) else start_time + (
                start_time - beat_times[i - 1])

        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)

        start_idx = min(start_idx, n_frames)
        end_idx = min(end_idx, n_frames)

        if end_idx > start_idx:
            segment = audio[start_idx:end_idx]
            energy = np.sqrt(np.mean(segment ** 2))
            energy_per_beat.append(energy)

    logger.info(f"Computed energy for {len(energy_per_beat)} beats")

    return torch.tensor(energy_per_beat)


def get_beat_segment(beat: int, metadata: Dict[str, Any]) -> str:
    """
    Get the segment label for a given beat.

    Args:
        beat: int, the beat index to check.
        metadata: dict, metadata containing segments and beats information.

    Returns:
        str: The label of the segment that contains the beat.
    """
    segments = metadata.get("segments")
    beats = metadata.get("beats")

    return (segment["label"] for segment in segments if segment["start"] <= beats[beat] < segment["end"])


def create_beat_graph(transitions: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Create a directed graph from the transitions' data.

    Args:
        transitions: List[Dict[str, Any]], the list of transitions where each transition contains a beat and its transitions.

    Returns:
        Dict[int, List[Dict[str, Any]]]: A dictionary where keys are beats and values are lists of transitions.
    """
    graph = defaultdict(list)

    for transition in transitions:
        beat = transition["beat"]

        for trans in transition["transitions"]:
            graph[beat].append({'to': trans["to"], 'dur': trans["duration"]})

    logger.info(f"Initialized graph with {len(graph)} beats.")

    return graph


def plot_similarity_matrix(
        similarity_matrix: torch.Tensor,
        save_path: str = "similarity_matrix.png",
        figsize: tuple = (24, 20),
        cmap: str = 'viridis',
        show_colorbar: bool = True,
        title: Optional[str] = None,
) -> None:
    """
    Plots the cosine similarity matrix and saves it to the specified path.
    Args:
        similarity_matrix: A 2D tensor representing the cosine similarity matrix of shape (num_beats, num_beats).
        save_path: a string specifying the path where the plot will be saved.
        figsize: a tuple specifying the size of the figure.
        cmap: a string specifying the colormap to use for the plot.
        show_colorbar: a boolean indicating whether to show the colorbar.
        title: an optional string specifying the title of the plot. If None, a default title will be used.

    Returns:
        None
    """
    if similarity_matrix.dim() != 2:
        raise ValueError("Similarity matrix should be a 2D tensor")

    matrix_np = similarity_matrix.detach().cpu().numpy()
    num_beats = matrix_np.shape[0]

    plt.figure(figsize=figsize)
    im = plt.imshow(matrix_np, cmap=cmap, aspect='auto', interpolation='nearest')

    if show_colorbar:
        cbar = plt.colorbar(im)
        cbar.set_label(f'Similarity Score', fontsize=32)

    if title is None:
        title = f'Cosine Similarity Matrix'

    plt.title(title, fontsize=32, pad=20)
    plt.xlabel('Beat Index', fontsize=32)
    plt.ylabel('Beat Index', fontsize=32)

    tick_interval = 20 if num_beats <= 100 else 10
    ticks = list(range(0, num_beats, tick_interval))
    labels = [str(i) for i in ticks]

    plt.xticks(
        ticks=ticks,
        labels=labels,
        rotation=90,
        fontsize=14
    )
    plt.yticks(
        ticks=ticks,
        labels=labels,
        fontsize=14
    )

    plt.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

    logger.info(f"Similarity matrix plot saved to {save_path}")
