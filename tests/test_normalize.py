import numpy as np
from preprocessing import normalize

def test_normalize_audio():
    # Sinal de áudio de exemplo
    audio = np.array([0.2, -0.5, 0.3, -0.1])
    normalized_audio = normalize.normalize_audio(audio)
    assert np.max(np.abs(normalized_audio)) == 1.0
    print("Teste de normalização passou com sucesso.")

if __name__ == "__main__":
    test_normalize_audio()
