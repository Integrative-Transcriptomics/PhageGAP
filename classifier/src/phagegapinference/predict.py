"""
Function class prediction utilities to load and apply `phagegapinference.model` within the `phagegapinference` package.

Currently, only :class:`phagegapinference.model.CNN` and :class:`phagegapinference.model.CNN_MLP` are supported.

The code is based on https://github.com/Integrative-Transcriptomics/PhageGAP-model/blob/5f9b63d160958ca5e96a1f0cc45be057d1fe5582/src/predict/predict.py.
"""

from __future__ import annotations

import torch, logging
import numpy as np
import pandas as pd
from typing import Dict
from torch.nn.utils.rnn import pad_sequence
from phagegapinference.model import CNN, CNN_MLP


# Initialize logging configuration.
logger = logging.getLogger(__name__)


def load_predict_model(path: str, num: int, model_type: str) -> tuple[CNN | CNN_MLP, dict, dict]:
	"""Loads the saved weights, sets the model to evaluation mode, and returns model metadata.

	Parameters
	__________
	path (str):
		Path to the saved model pickle file.
	num (int):
		Model number to load (e.g., 1, 2, etc.).
	model_type (str):
		Type of model to load (`cnn` or `cnn_mlp`).

	Returns
	_______
	model (torch.nn.Module):
		The loaded PyTorch model with weights restored.
	label_map (dict):
		Mapping from predicted class indices to human-readable labels. If the checkpoint
		stores labels as {label: index}, this mapping is inverted to {index: label}.
	cfg (dict):
		The configuration dictionary used to initialize the model, taken from the checkpoint.
	"""
	# TODO: Why is map_location="cpu" fixed? Shouldn't it be device agnostic?
	checkpoint = torch.load(path, map_location="cpu", weights_only=False)
	cfg = checkpoint["config"]
	label_map = checkpoint["label_map"]

	# TODO: Adjust default parameters to prevent failure when loading models with incomplete configurations.
	if model_type == "cnn":
		model = CNN(
			trial=None,
			in_channels=cfg["data"]["embed_dim"],
			num_conv_layers=cfg["models"][f"model{num}"]["cnn"]["num_conv_layers"],
			num_filters=cfg["models"][f"model{num}"]["cnn"]["num_filters"],
			kernel_sizes=cfg["models"][f"model{num}"]["cnn"]["kernel_sizes"],
			dropout=cfg["models"][f"model{num}"]["cnn"]["dropout"],
			num_classes=len(label_map),
			dilations=cfg["models"][f"model{num}"]["cnn"]["dilations"],
			use_dilation=cfg["models"][f"model{num}"]["cnn"]["use_dilation"],
			mean_max=cfg["models"][f"model{num}"]["cnn"].get("mean_max", False),
			n_feats=0,
			use_linear_attention=cfg["models"][f"model{num}"]["cnn"].get("use_linear_attention", False),
			use_nonlinear_attention=cfg["models"][f"model{num}"]["cnn"].get("use_nonlinear_attention", False)
		)
	elif model_type == "cnn_mlp":
		model = CNN_MLP(
			trial=None,
			in_channels=cfg["data"]["embed_dim"],
			num_conv_layers=cfg["models"][f"model{num}"]["cnn_mlp"]["num_conv_layers"],
			num_filters=cfg["models"][f"model{num}"]["cnn_mlp"]["num_filters"],
			kernel_sizes=cfg["models"][f"model{num}"]["cnn_mlp"]["kernel_sizes"],
			dropout_conv=cfg["models"][f"model{num}"]["cnn_mlp"]["dropout_conv"],
			num_classes=len(label_map),
			dilations=cfg["models"][f"model{num}"]["cnn_mlp"]["dilations"],
			use_dilation=cfg["models"][f"model{num}"]["cnn_mlp"]["use_dilation"],
			use_linear_attention=cfg["models"][f"model{num}"]["cnn_mlp"]["use_linear_attention"],
			use_nonlinear_attention=cfg["models"][f"model{num}"]["cnn_mlp"]["use_nonlinear_attention"],
			num_dimensions=cfg["models"][f"model{num}"]["cnn_mlp"]["num_dimensions"],
			num_neurons=cfg["models"][f"model{num}"]["cnn_mlp"]["num_neurons"],
			dropout_mlp=cfg["models"][f"model{num}"]["cnn_mlp"]["dropout_mlp"],
		)
	else:
		raise ValueError(f"Unsupported model type: {model_type}. Supported types are 'cnn' and 'cnn_mlp'.")

	model.load_state_dict(checkpoint["state_dict"])
	model.eval()

	# exchange keys and values for convenience
	label_map = {int(v): str(k) for k, v in label_map.items()}

	return model, label_map, cfg


def run_prediction(embeddings: Dict[str, np.ndarray], metadata_df: pd.DataFrame, predict_model: CNN | CNN_MLP, label_map: Dict[int, str]) -> pd.DataFrame:
	"""Predict classes for protein embeddings using the selected classifier model.

	Parameters
	__________
	embeddings (Dict[str, np.ndarray]):
		Dictionary of protein sequence embeddings {protein_ID: embedding}.
	metadata_df (pd.DataFrame):
		DataFrame with metadata.
	predict_model (CNN | CNN_MLP):
		Instance of CNN or CNN_MLP to use for class predictions.
	label_map (Dict[int, str]):
		The label map of the specified classifier.

	Returns
	_______
	results (pd.DataFrame):
		DataFrame containing predictions and probabilities.
	"""
	assert type(predict_model) in (CNN, CNN_MLP)

	protein_ids = list(embeddings.keys())

	# Parameters for batch prediction.
	batch_size = 32
	all_probs = []
	all_features = []

	for i in range(0, len(protein_ids), batch_size):
		batch_ids = protein_ids[i:i+batch_size]
		tensor_list = [torch.from_numpy(embeddings[pid]).float() for pid in batch_ids]
		X = pad_sequence(tensor_list, batch_first=True) # shape (batch, max_len, D)
		lengths = torch.tensor([t.shape[0] for t in tensor_list])
		mask = torch.arange(X.size(1))[None, :] < lengths[:, None]

		# Forward pass.
		with torch.no_grad():
			logits, features = predict_model(X, mask, features=None, return_features=True)
			probs = torch.softmax(logits, dim=-1)

		all_features.append(features.cpu())
		all_probs.append(probs)

	probs = torch.cat(all_probs).numpy()
	all_features = torch.cat(all_features)
	filter_dict = {pid: feat.numpy() for pid, feat in zip(protein_ids, all_features)}

	# For each row, get sorted descending indices.
	topk = np.argsort(probs, axis=1)[:, ::-1]

	top1_idx = topk[:, 0]
	top2_idx = topk[:, 1]
	top3_idx = topk[:, 2]

	top1_prob = probs[np.arange(len(probs)), top1_idx]
	top2_prob = probs[np.arange(len(probs)), top2_idx]
	top3_prob = probs[np.arange(len(probs)), top3_idx]

	top1_label = [label_map[int(i)] for i in top1_idx]
	top2_label = [label_map[int(i)] for i in top2_idx]
	top3_label = [label_map[int(i)] for i in top3_idx]

	protein_ids = list(embeddings.keys())

	results = pd.DataFrame({
		"protein_ID": protein_ids,
		"top1": top1_label,
		"top2": top2_label,
		"top3": top3_label,
		"P(top1)": top1_prob,
		"P(top2)": top2_prob,
		"P(top3)": top3_prob
	})

	results = pd.merge(metadata_df, results, on="protein_ID", how="left")

	return results, filter_dict
