import os
from datetime import datetime
from safetensors.torch import save_file, load_file


def save_model(model, name: str, folder: str = "models"):
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{name}_{timestamp}.safetensors"
    path = os.path.join(folder, filename)

    save_file(model.state_dict(), path)
    print(f"Saved model to {path}")
    return path


def load_model(model, name: str, folder: str = "models"):
    path = os.path.join(folder, name)
    state_dict = load_file(path)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {path}")
    return model
