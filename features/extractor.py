import numpy as np

def extract_features(vocalizations, spectrogram, sr, hop_length):
    """
    Extrai características de cada vocalização detectada.

    Args:
        vocalizations (list): Lista de dicionários com informações das vocalizações.
        spectrogram (np.ndarray): Espectrograma em escala de decibéis.
        sr (int): Taxa de amostragem do áudio.
        hop_length (int): Passo entre janelas na STFT.

    Returns:
        List[dict]: Lista de dicionários contendo características extraídas de cada vocalização.
    """
    features = []
    for v in vocalizations:
        start_frame = int(v['start_time'] * sr / hop_length)
        end_frame = start_frame + int(v['duration'] * sr / hop_length)
        spec_segment = spectrogram[:, start_frame:end_frame]

        # Cálculo das características
        duration = v['duration']
        min_freq = v['min_freq']
        max_freq = v['max_freq']
        mean_freq = (min_freq + max_freq) / 2
        intensity = np.mean(spec_segment)
        entropy = -np.sum(spec_segment * np.log2(spec_segment + 1e-10))

        features.append({
            'duration': duration,
            'min_freq': min_freq,
            'max_freq': max_freq,
            'mean_freq': mean_freq,
            'intensity': intensity,
            'entropy': entropy
        })

    return features
