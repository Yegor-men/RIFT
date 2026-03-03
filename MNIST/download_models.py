# Download the pretrained models for diffusion on MNIST
import os
from huggingface_hub import hf_hub_download

REPO_ID = "yegor-men/resolution-invariant-image-diffuser"
R2IR_filename = "MNIST_R2IR.safetensors"
R2ID_filename = "MNIST_R2ID.safetensors"
TEXT_filename = "MNIST_TEXT.safetensors"

LOCAL_DIR = "models"

os.makedirs(LOCAL_DIR, exist_ok=True)

for filename in (R2IR_filename, R2ID_filename, TEXT_filename):
    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=LOCAL_DIR,
    )

    print(f"{filename} file saved to: {os.path.join(LOCAL_DIR, filename)}")
