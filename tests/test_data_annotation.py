import os
from classification.train import train_model
from classification.predict import predict

def test_classification_pipeline():
    annotations_file = 'data/annotations.csv'
    audio_dir = 'data/Audios'

    # Treinar o modelo
    train_model(annotations_file, audio_dir, num_epochs=1, batch_size=2)

    # Testar a predição
    test_image_path = 'data/test_spectrogram.png'
    predicted_label = predict(test_image_path)
    assert isinstance(predicted_label, int), "A predição deve retornar um inteiro representando a classe."
