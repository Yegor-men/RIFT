import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.inference import render_checkpoint_samples


# CONFIG ===============================================================================================================

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = None  # If None, use newest MNIST_E*_rift.safetensors from MODEL_DIR by file modification time.
CONDITIONER_PATH = None  # If None, inferred from MODEL_PATH.
CONFIG_PATH = None       # If None, inferred from MODEL_PATH.

SIZES = (28, 64, 128)
LABELS = "grid"
BATCH_SIZE = 100
SAMPLE_STEPS = 20
STEP_SIZE = 0.05
INVERT_STEPS = 0
CONDITION_STRENGTH = 1.0
EVIDENCE_SCALE = 1.0
DEVICE = "cuda"
SAVE_IMAGES = False


# INFERENCE ============================================================================================================

def main() -> None:
    render_checkpoint_samples(
        model_path=MODEL_PATH,
        model_dir=MODEL_DIR,
        dataset_name="MNIST",
        conditioner_path=CONDITIONER_PATH,
        config_path=CONFIG_PATH,
        title_prefix="MNIST",
        sizes=SIZES,
        labels=LABELS,
        batch_size=BATCH_SIZE,
        sample_steps=SAMPLE_STEPS,
        step_size=STEP_SIZE,
        invert_steps=INVERT_STEPS,
        condition_strength=CONDITION_STRENGTH,
        evidence_scale=EVIDENCE_SCALE,
        device=DEVICE,
        save=SAVE_IMAGES,
    )


if __name__ == "__main__":
    main()
