# VocalMat-Py

A Python implementation of VocalMat, a tool for analyzing ultrasonic vocalizations from mice using computer vision and machine learning.

## Description

VocalMat-Py is a Python port of the original VocalMat MATLAB implementation by the Dietrich Lab. This project aims to provide the same functionality as the original VocalMat but in a Python environment, making it more accessible and easier to integrate with modern machine learning workflows.

The original VocalMat was developed by Fonseca et al. and is available at [www.dietrich-lab.org/vocalmat](https://www.dietrich-lab.org/vocalmat).

## Citation

If you use this software in your research, please cite the original VocalMat paper:

```bibtex
@article{Fonseca2021AnalysisOU,
  title={Analysis of ultrasonic vocalizations from mice using computer vision and machine learning},
  author={Antonio H. O. Fonseca and Gustavo Madeira Santana and Gabriela M Bosque Ortiz and Sergio Bampi and Marcelo O. Dietrich},
  journal={eLife},
  year={2021},
  volume={10}
}
```

## Features

- Audio loading and processing
- Signal filtering and preprocessing
- Feature extraction
- Audio segmentation
- Classification of ultrasonic vocalizations
- Preprocessing utilities

## Installation

The package requires Python 3.12 or higher. You can install it using uv:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package
uv pip install vocalmat-py
```

## Dependencies

- librosa >= 0.11.0
- matplotlib >= 3.10.3
- Additional dependencies listed in pyproject.toml

## Project Structure

```
vocalmat-py/
├── audio/           # Audio processing modules
├── classification/  # Classification models and utilities
├── data/           # Data storage and management
├── features/       # Feature extraction
├── preprocessing/  # Preprocessing utilities
├── segmentation/   # Audio segmentation tools
├── utils/          # Utility functions
├── tests/          # Test suite
├── notebooks/      # Example notebooks
└── scripts/        # Command-line scripts
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
2. Create and activate a virtual environment
3. Install development dependencies using uv
4. Run tests

```bash
git clone https://github.com/yourusername/vocalmat-py.git
cd vocalmat-py
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
pytest
```

## License

This project is licensed under the terms of the license included in the LICENSE file. This is a derivative work of the original VocalMat project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing, please ensure that your code follows the project's coding standards and includes appropriate tests.

## Contact

For questions and support, please open an issue in the GitHub repository.

## Acknowledgments

This project is a Python port of the original VocalMat MATLAB implementation by the Dietrich Lab. We thank the original authors for their work and for making their code available to the scientific community.
