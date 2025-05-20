import logging
import sys
import torch
import warnings
from tqdm import tqdm
from typing import Optional

# Configurar filtros de warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.models._utils')
warnings.filterwarnings('ignore', category=UserWarning, module='librosa.feature.spectral')

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura um logger com formato padrão.
    
    Args:
        name: Nome do logger
        log_file: Arquivo opcional para salvar os logs
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Formato padrão
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (se especificado)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_device_info() -> str:
    """
    Retorna informações sobre o dispositivo disponível (CPU/GPU).
    """
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        return f"GPU: {device}"
    return "CPU"

def create_progress_bar(iterable, desc: str, total: Optional[int] = None) -> tqdm:
    """
    Cria uma barra de progresso com formato padrão.
    
    Args:
        iterable: Iterável para a barra de progresso
        desc: Descrição da operação
        total: Total de iterações (opcional)
    
    Returns:
        Barra de progresso configurada
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ) 