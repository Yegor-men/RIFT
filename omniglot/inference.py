import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.inference import render_checkpoint_samples


# CONFIG ===============================================================================================================

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = None  # If None, use newest Omniglot_E*_r2id.safetensors from MODEL_DIR by file modification time.
CONDITIONER_PATH = None  # If None, inferred from MODEL_PATH.
CONFIG_PATH = None       # If None, inferred from MODEL_PATH.

SIZES = (32, 64, 128)
LABELS = "grid"
BATCH_SIZE = 100
SAMPLE_STEPS = 100
STEP_SIZE = 1.0
CONDITION_STRENGTH = 1.0
EVIDENCE_SCALE = 1.0
DEVICE = "cuda"
SAVE_IMAGES = False


# INFERENCE ============================================================================================================

def main() -> None:
    render_checkpoint_samples(
        model_path=MODEL_PATH,
        model_dir=MODEL_DIR,
        dataset_name="Omniglot",
        conditioner_path=CONDITIONER_PATH,
        config_path=CONFIG_PATH,
        title_prefix="Omniglot",
        sizes=SIZES,
        labels=LABELS,
        batch_size=BATCH_SIZE,
        sample_steps=SAMPLE_STEPS,
        step_size=STEP_SIZE,
        condition_strength=CONDITION_STRENGTH,
        evidence_scale=EVIDENCE_SCALE,
        device=DEVICE,
        save=SAVE_IMAGES,
    )


if __name__ == "__main__":
    main()
