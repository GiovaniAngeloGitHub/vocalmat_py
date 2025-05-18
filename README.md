# VocalMat-Py

A Python library for audio processing and analysis, focusing on vocal material processing and feature extraction.

## Description

VocalMat-Py is a Python package designed for audio processing, feature extraction, and analysis of vocal materials. It provides tools for audio loading, filtering, preprocessing, segmentation, and feature extraction.

## Features

- Audio loading and processing
- Signal filtering
- Feature extraction
- Audio segmentation
- Preprocessing utilities

## Installation

The package requires Python 3.12 or higher. You can install it using pip:

```bash
pip install vocalmat-py
```

## Dependencies

- librosa >= 0.11.0
- matplotlib >= 3.10.3

## Project Structure

```
vocalmat-py/
├── audio/           # Audio processing modules
│   ├── filter.py    # Signal filtering
│   └── loader.py    # Audio loading utilities
├── features/        # Feature extraction
├── preprocessing/   # Preprocessing utilities
├── segmentation/    # Audio segmentation tools
├── utils/          # Utility functions
├── tests/          # Test suite
└── notebooks/      # Example notebooks
```

## Usage

```python
from vocalmat_py.audio.loader import load_audio
from vocalmat_py.audio.filter import apply_filter

# Load audio file
audio_data = load_audio("path/to/audio.wav")

# Apply processing
processed_audio = apply_filter(audio_data)
```

## Development

To set up the development environment:

1. Clone the repository
2. Install development dependencies
3. Run tests

```bash
git clone https://github.com/yourusername/vocalmat-py.git
cd vocalmat-py
pip install -e .
pytest
```

## License

This project is licensed under the terms of the license included in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue in the GitHub repository.
