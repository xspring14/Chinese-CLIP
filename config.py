import pathlib

import torch
import cn_clip.clip as clip

APP_DIR = pathlib.Path(__file__).parent.absolute()

# Next set the project root directory.
#PROJECT_DIR = APP_DIR.parent

# Set the model directory & create if it doesn't exist
MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Clip specific variables
MAX_TEXT_LENGTH = 4096
DEFAULT_CLIP_MODEL = "RN50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AVAILABLE_MODELS = clip.available_models()
