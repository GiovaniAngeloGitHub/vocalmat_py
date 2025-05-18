import numpy as np
from skimage import measure

def filter_regions(labeled_regions, min_area=100, max_area=10000, min_eccentricity=0.0, max_eccentricity=1.0):
    """
    Filtra regiões rotuladas com base em propriedades geométricas.

    Args:
        labeled_regions (np.ndarray): Matriz com regiões rotuladas.
        min_area (int): Área mínima das regiões a serem mantidas.
        max_area (int): Área máxima das regiões a serem mantidas.
        min_eccentricity (float): Excentricidade mínima das regiões a serem mantidas.
        max_eccentricity (float): Excentricidade máxima das regiões a serem mantidas.

    Returns:
        filtered_labels (np.ndarray): Matriz com regiões filtradas.
        filtered_props (list): Lista de propriedades das regiões filtradas.
    """
    # Obter propriedades das regiões
    props = measure.regionprops(labeled_regions)

    # Inicializar matriz para regiões filtradas
    filtered_labels = np.zeros_like(labeled_regions)
    filtered_props = []

    for prop in props:
        if min_area <= prop.area <= max_area and min_eccentricity <= prop.eccentricity <= max_eccentricity:
            # Manter região
            filtered_labels[labeled_regions == prop.label] = prop.label
            filtered_props.append(prop)

    return filtered_labels, filtered_props
