import os
import numpy as np
import librosa
from audio import loader, spectrogram
from preprocessing import threshold
from segmentation import morphology, post_filter

def test_segmentation_with_stream():
    # Caminho para o arquivo de áudio
    audio_path = 'data/Audios/1303.WAV'
    
    # Parâmetros para o espectrograma
    frame_length = 4096
    hop_length = 2048
    block_length = 512

    # Obter a taxa de amostragem do arquivo
    sr = librosa.get_samplerate(audio_path)

    # Criar gerador de blocos de áudio
    stream = librosa.stream(
        audio_path,
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
        binary_spectrogram = threshold.apply_threshold(S_db_block)
        # Segmentar regiões
        labeled_regions, num_regions = morphology.segment_regions(binary_spectrogram)
        # Aplicar pós-filtragem
        filtered_labels, filtered_props = post_filter.filter_regions(labeled_regions)

        # Verificações simples
        assert labeled_regions.shape == binary_spectrogram.shape, "Dimensões incompatíveis entre regiões rotuladas e espectrograma binarizado."
        assert num_regions >= 0, "Número de regiões deve ser não-negativo."
        assert filtered_labels.shape == labeled_regions.shape, "Dimensões incompatíveis após pós-filtragem."
        break  # Remova este break para processar todos os blocos
