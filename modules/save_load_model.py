import os
import torch
from datetime import datetime


def save_checkpoint(model: torch.nn.Module, folder: str = "models", prefix: str = "s2ir"):
	"""
	Save model.state_dict() into `folder` with a timestamped filename.
	Returns the saved path.
	"""
	os.makedirs(folder, exist_ok=True)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	fname = f"{prefix}_{ts}.pt"
	path = os.path.join(folder, fname)
	torch.save(model.state_dict(), path)
	print(f"Saved checkpoint: {path}")
	return path


def load_checkpoint_into(model: torch.nn.Module, path: str, device=None):
	"""
	Load a state dict into an existing model instance.
	Model must be already instantiated with the correct config.
	"""
	map_location = device if device is not None else torch.device("cpu")
	sd = torch.load(path, map_location=map_location)
	model.load_state_dict(sd)
	print(f"Loaded checkpoint from: {path}")
	return model
