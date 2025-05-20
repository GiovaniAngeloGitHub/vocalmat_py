import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.threshold import apply_threshold
from utils.logger import setup_logger, get_device_info, create_progress_bar

logger = setup_logger('spectrogram', 'logs/spectrogram.log')
logger.info(f"Using device: {get_device_info()}")

def generate_spectrogram_stream(file_path, frame_length=4096, hop_length=2048, block_length=512, threshold_db=-50.0):
    """
    Gera e exibe o espectrograma de um arquivo de áudio utilizando processamento em blocos,
    aplicando limiarização a cada bloco.

    Args:
        file_path (str): Caminho para o arquivo de áudio.
        frame_length (int): Tamanho da janela para STFT.
        hop_length (int): Passo entre janelas.
        block_length (int): Número de frames por bloco.
        threshold_db (float): Valor de limiar em decibéis.
    """
    # Obter a taxa de amostragem do arquivo
    sr = librosa.get_samplerate(file_path)
    logger.info(f"Processing audio file: {file_path} (Sample rate: {sr}Hz)")

    # Ajustar parâmetros para evitar filtros mel vazios
    n_mels = 128  # Reduzir o número de filtros mel
    fmax = sr/2  # Frequência máxima = metade da taxa de amostragem

    # Criar gerador de blocos de áudio
    stream = librosa.stream(
        file_path,
        block_length=block_length,
        frame_length=frame_length,
        hop_length=hop_length,
        mono=True,
        dtype=np.float32
    )

    # Calcular número total de blocos para a barra de progresso
    audio_duration = librosa.get_duration(path=file_path)
    total_blocks = int(np.ceil(audio_duration * sr / (hop_length * block_length)))
    
    for y_block in create_progress_bar(stream, desc="Generating spectrograms", total=total_blocks):
        # Calcular o STFT do bloco
        D_block = librosa.stft(y_block, n_fft=frame_length, hop_length=hop_length, center=False)
        # Converter para escala de decibéis
        S_db_block = librosa.amplitude_to_db(np.abs(D_block), ref=np.max)
        # Aplicar limiarização
        mask = apply_threshold(S_db_block, threshold_db)
        # Visualizar a máscara binária
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mask, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.colorbar()
        plt.title('Máscara Binária do Espectrograma')
        plt.tight_layout()
        plt.show()
