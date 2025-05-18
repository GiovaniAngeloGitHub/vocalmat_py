import matplotlib.pyplot as plt

from audio import filter, loader

# Carregar e normalizar o áudio
file_path = "data/Audios/1303.WAV"
audio, sr = loader.load_audio(file_path)
audio_norm = loader.normalize_audio(audio)

# Aplicar o filtro passa-alta
audio_filtered = filter.highpass_filter(audio_norm, sr, cutoff=20000.0, order=5)

# Plotar os sinais original e filtrado
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(audio_norm)
plt.title("Áudio Normalizado")
plt.subplot(2, 1, 2)
plt.plot(audio_filtered)
plt.title("Áudio Após Filtro Passa-Alta")
plt.tight_layout()
plt.show()
