import numpy as np
from scipy.signal import butter, filtfilt


def highpass_filter(
    audio: np.ndarray, sr: int, cutoff: float = 20000.0, order: int = 5
) -> np.ndarray:
    """
    Aplica um filtro passa-alta Butterworth ao sinal de áudio.

    Args:
        audio (np.ndarray): Sinal de áudio.
        sr (int): Taxa de amostragem do áudio (Hz).
        cutoff (float): Frequência de corte do filtro (Hz).
        order (int): Ordem do filtro.

    Returns:
        np.ndarray: Sinal de áudio filtrado.
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio
