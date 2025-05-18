import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.threshold import apply_threshold

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

    # Criar gerador de blocos de áudio
    stream = librosa.stream(
        file_path,
        block_length=block_length,
        frame_length=frame_length,
        hop_length=hop_length,
        mono=True,
        dtype=np.float32
    )

    for y_block in stream:
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
