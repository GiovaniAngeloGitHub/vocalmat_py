import numpy as np
import librosa
import pytest
from preprocessing import threshold

def test_apply_threshold_on_spectrogram_block():
    # Carregar um arquivo de áudio para teste
    file_path = 'data/Audios/1303.WAV'
    sr = librosa.get_samplerate(file_path)
    
    # Parâmetros para STFT
    frame_length = 4096
    hop_length = 2048
    block_length = 512

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
        binary_mask = threshold.apply_threshold(S_db_block, threshold_db=-40.0)
        
        # Verificar se a máscara binária tem a mesma forma do espectrograma
        assert binary_mask.shape == S_db_block.shape, "A máscara binária deve ter a mesma forma do espectrograma."
        
        # Verificar se a máscara contém apenas valores 0 e 1
        unique_values = np.unique(binary_mask)
        assert set(unique_values).issubset({0, 1}), "A máscara binária deve conter apenas 0 e 1."
        
        # Opcional: verificar se há pelo menos alguns valores positivos na máscara
        assert np.any(binary_mask == 1), "A máscara binária deve conter pelo menos alguns valores 1."
        
        # Testar apenas o primeiro bloco para evitar longos tempos de execução
        break
