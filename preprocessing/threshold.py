import numpy as np

def apply_threshold(S_db_block: np.ndarray, threshold_db: float = -50.0) -> np.ndarray:
    """
    Aplica limiarização a um bloco de espectrograma em decibéis.

    Args:
        S_db_block (np.ndarray): Bloco do espectrograma em decibéis.
        threshold_db (float): Valor de limiar em decibéis.

    Returns:
        np.ndarray: Máscara binária do espectrograma após limiarização.
    """
    # Cria uma máscara onde os valores acima do limiar são 1, caso contrário 0
    mask = S_db_block > threshold_db
    return mask.astype(np.uint8)
