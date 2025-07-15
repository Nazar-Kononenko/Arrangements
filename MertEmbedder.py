import numpy as np
import logging
import torch
import torchaudio.transforms as T

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
from datetime import datetime


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MERTBeatEmbedder:
    FRAME_RATE = 75.0

    def __init__(self, model_name: str, chunk_duration: float = 120.0):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.chunk_duration = chunk_duration

    def get_audio_embeddings(self, audio: np.ndarray, sr: int):
        resample_rate = self.processor.sampling_rate

        if resample_rate != sr:
            logger.info(f'setting rate from {sr} to {resample_rate}')
            resampler = T.Resample(sr, resample_rate)
        else:
            resampler = None

        if resampler is None:
            input_audio = audio
        else:
            input_audio = resampler(torch.from_numpy(audio))

        inputs = self.processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")

        if self.model.device.type == 'cuda':
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        start_time = datetime.now()
        with torch.no_grad():
            output = self.model(**inputs)
        end_time = datetime.now()

        logger.info(f"Embedding model inference time: {end_time - start_time}")

        return output.last_hidden_state.squeeze(0).detach().cpu()

    def _split_audio(self, audio: np.ndarray, sr: int) -> list:
        samples_per_chunk = int(self.chunk_duration * sr)
        audio_length = len(audio)
        chunks = []

        for start in range(0, audio_length, samples_per_chunk):
            end = min(start + samples_per_chunk, audio_length)
            chunks.append(audio[start:end])

        return chunks

    def __call__(self, audio: np.ndarray, sr: int, audio_metadata: dict) -> torch.Tensor:
        beat_times = audio_metadata.get("beats")
        audio_embeddings = self.get_audio_embeddings(audio, sr)

        if not isinstance(beat_times, list) or len(beat_times) < 2:
            raise ValueError("beat_times should be a list with at least two elements")

        n_frames, _ = audio_embeddings.shape

        beat_embeddings = []

        for i in range(len(beat_times)):
            start_time = beat_times[i]
            end_time = beat_times[i + 1] if i + 1 < len(beat_times) else start_time + (
                    start_time - beat_times[i - 1])

            start_idx = int(start_time * self.FRAME_RATE)
            end_idx = int(end_time * self.FRAME_RATE)

            start_idx = min(start_idx, n_frames)
            end_idx = min(end_idx, n_frames)

            if end_idx > start_idx:
                emb = audio_embeddings[start_idx:end_idx].mean(dim=0)
                beat_embeddings.append(emb)

        logger.info(f"Accumulated {len(beat_embeddings)} beat embeddings")

        return torch.stack(beat_embeddings)

    def to(self, device: str):
        self.model.to(device)
        return self
