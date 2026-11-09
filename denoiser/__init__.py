"""Traditional (non-learning) speech noise reduction for the Noise Reduction Challenge."""

from .config import CONFIG, Config
from .pipeline import denoise_signal, denoise_file, denoise_folder

__all__ = ["CONFIG", "Config", "denoise_signal", "denoise_file", "denoise_folder"]
__version__ = "0.1.0"
