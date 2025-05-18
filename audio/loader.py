import os
import soundfile as sf
import numpy as np
from typing import Tuple


def load_audio(file_path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
    """
    Carrega um arquivo de áudio WAV mono.

    Args:
        file_path (str): Caminho para o arquivo de áudio.
        target_sr (int, opcional): Frequência de amostragem desejada. Se None, mantém a original.

    Returns:
        Tuple[np.ndarray, int]: O sinal de áudio e sua taxa de amostragem (Hz).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    audio, sr = sf.read(file_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Converte para mono se necessário

    if target_sr and sr != target_sr:
        from librosa import resample
        audio = resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normaliza o áudio para ter amplitude máxima igual a 1.0.

    Args:
        audio (np.ndarray): O sinal de áudio bruto.

    Returns:
        np.ndarray: O sinal de áudio normalizado.
    """
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val
