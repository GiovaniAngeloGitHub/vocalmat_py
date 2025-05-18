import numpy as np

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normaliza o sinal de áudio para que a amplitude máxima seja 1.0.

    Args:
        audio (np.ndarray): Sinal de áudio.

    Returns:
        np.ndarray: Sinal de áudio normalizado.
    """
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude == 0:
        return audio
    return audio / max_amplitude
