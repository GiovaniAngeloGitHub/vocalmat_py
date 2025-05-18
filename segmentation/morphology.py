import numpy as np
from skimage import morphology, measure

def segment_regions(binary_spectrogram, min_size=50, connectivity=2):
    """
    Segmenta regiões no espectrograma binarizado usando operações morfológicas.

    Args:
        binary_spectrogram (np.ndarray): Espectrograma binarizado (valores 0 e 1).
        min_size (int): Tamanho mínimo das regiões a serem mantidas.
        connectivity (int): Conectividade para rotulagem (1 ou 2).

    Returns:
        labeled_regions (np.ndarray): Matriz com regiões rotuladas.
        num_regions (int): Número de regiões detectadas.
    """
    # Remover pequenas regiões
    cleaned = morphology.remove_small_objects(binary_spectrogram.astype(bool), min_size=min_size, connectivity=connectivity)

    # Preencher pequenos buracos
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size, connectivity=connectivity)

    # Rotular regiões conectadas
    labeled_regions = measure.label(cleaned, connectivity=connectivity)

    num_regions = np.max(labeled_regions)

    return labeled_regions, num_regions
