# Download pretrained raw-pixel R2ID MNIST checkpoints.
import os
from huggingface_hub import hf_hub_download

REPO_ID = "yegor-men/resolution-invariant-image-diffuser"
MODEL_FILENAME = "MNIST_R2ID.safetensors"
CONDITIONER_FILENAME = "MNIST_CONDITIONER.safetensors"
CONFIG_FILENAME = "MNIST_CONFIG.json"
LOCAL_DIR = "models"

os.makedirs(LOCAL_DIR, exist_ok=True)

for filename in (MODEL_FILENAME, CONDITIONER_FILENAME, CONFIG_FILENAME):
    hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=LOCAL_DIR)
    print(f"{filename} file saved to: {os.path.join(LOCAL_DIR, filename)}")
