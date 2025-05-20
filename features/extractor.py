import numpy as np
from utils.logger import setup_logger, get_device_info, create_progress_bar

logger = setup_logger('feature_extractor', 'logs/feature_extraction.log')
logger.info(f"Using device: {get_device_info()}")

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
    logger.info("Starting feature extraction")
    
    features = []
    pbar = create_progress_bar(range(len(vocalizations)), desc="Extracting features")
    
    for i in pbar:
        v = vocalizations[i]
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
        pbar.set_postfix({'sample': f'{i+1}/{len(vocalizations)}'})
    
    logger.info("Feature extraction completed")
    return features
