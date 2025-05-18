# Caminho para o arquivo de áudio
from audio import loader

file_path = "data/Audios/1303.WAV"

# Carregar o áudio
audio, sr = loader.load_audio(file_path)

# Normalizar o áudio
audio_norm = loader.normalize_audio(audio)

# Exibir informações básicas
print(f"Taxa de amostragem: {sr} Hz")
print(f"Duração do áudio: {len(audio) / sr:.2f} segundos")
print(f"Amplitude máxima após normalização: {max(audio_norm):.2f}")
