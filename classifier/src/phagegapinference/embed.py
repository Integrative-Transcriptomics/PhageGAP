"""
pLM model implementations of the `phagegapinference` package.

Currently only ProtT5 is supported. Other models are not yet implemented. If you want
to re-add support for other models, please see the `embed.py` file in the `deprecated`
folder of the `phagegap` code repository.

The code is based on https://github.com/Integrative-Transcriptomics/PhageGAP-model/blob/5f9b63d160958ca5e96a1f0cc45be057d1fe5582/src/embed/embed.py.
"""

from __future__ import annotations

import logging
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any


# Initialize logging configuration.
logger = logging.getLogger(__name__)


# Define a type alias for model configuration dictionaries.
ModelConfig = Dict[str, Any]


# Define a dictionary containing configurations for different protein language models (pLMs).
# Adapted from https://github.com/tsenoner/plm_choice/blob/main/src/data_preparation/embeddings/embedding_generation.py
MODEL_CONFIGS: Dict[str, ModelConfig] = {
	"prot_t5": {
		"hf_id": "Rostlab/prot_t5_xl_half_uniref50-enc",
		"loader": "transformers",
		"model_class": T5EncoderModel,
		"tokenizer_class": T5Tokenizer,
		"load_kwargs": {"torch_dtype": torch.float16, "device_map": "auto",},
		"post_load_hook": lambda m: m.half() if hasattr(m, "half") else m,
		"tokenizer_load_kwargs": {"do_lower_case": False},
	},
	"prot_t5xxl": {
		"hf_id": "Rostlab/prot_t5_xxl_uniref50",
		"loader": "transformers",
		"model_class": T5EncoderModel,
		"tokenizer_class": T5Tokenizer,
		"tokenizer_load_kwargs": {"do_lower_case": False},
	},
	"prost_t5": {
		"hf_id": "Rostlab/ProstT5_fp16",
		"loader": "transformers",
		"model_class": T5EncoderModel,
		"tokenizer_class": T5Tokenizer,
		"load_kwargs": {"torch_dtype": torch.float16},
		"post_load_hook": lambda m: m.half(),
		"tokenizer_load_kwargs": {"do_lower_case": False},
	}
}


def preprocess_df(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
	"""Preprocess protein sequences for embedding depending on which model type is chosen.

	Parameters
	__________
	df (pd.DataFrame):
		DataFrame containing protein sequences and metadata. Must contain a column "protein_seq".
		See :func:`phagegapinference.utils.parse_sequence_data` for expected format.
	model_type (str):
		Type of model to use for embedding. Options: "prot_t5", "prost_t5", "prot_t5xxl".

	Returns
	_______
	pd.DataFrame:
		DataFrame with an additional column "processed_seq" containing preprocessed sequences ready for embedding.
	"""
	if not "protein_seq" in df.columns:
		raise ValueError("Input DataFrame must contain a 'protein_seq' column with protein sequences.")

	df["protein_seq"] = df["protein_seq"].str.upper()
	df["processed_seq"] = df["protein_seq"].str.replace("[UZOB]", "X", regex=True)

	if model_type in ("prot_t5", "prot_t5xxl"):
		# Only add spaces.
		df["processed_seq"] = df["processed_seq"].apply(lambda aa: " ".join(list(aa)))
	elif model_type == "prost_t5":
		# Add spaces and prefix for ProstT5.
		df["processed_seq"] = df["processed_seq"].apply(lambda aa: "<AA2fold> " + " ".join(list(aa)))
	else:
		logger.warning(f"Model type {model_type} not recognized for preprocessing. No preprocessing applied.")

	return df


def load_embed_model(model_type: str, device: str) -> Tuple[nn.Module, Optional[T5Tokenizer]]:
	"""Sets up the pLM model.

	Parameters
	__________
	model_type (str):
		Type of model to use for embedding. Options: "prot_t5", "prost_t5", "prot_t5xxl".
	device (str):
		Device to load the model onto. Options: "cpu", "cuda".
	
	Returns
	_______
	tuple[nn.Module, Optional[T5Tokenizer]]:
		The loaded model and its tokenizer (if applicable). The tokenizer is required for ProtT5, ProstT5, and gLM2 models.
		For other models, the tokenizer may be None.

	Raises
	______
	AssertionError 
		If model type or loader is not supported.
	RunTimeError
		If model could not be loaded.
	"""
	if model_type not in MODEL_CONFIGS:
		raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(MODEL_CONFIGS.keys())}")
	# Load configuration for the specified checkpoint/model.
	config = MODEL_CONFIGS[model_type]
	hf_id = config["hf_id"]
	loader = config["loader"]
	model_class = config["model_class"]
	tokenizer_class = config["tokenizer_class"]
	load_kwargs = config.get("load_kwargs", {})
	post_load_hook = config.get("post_load_hook")
	tokenizer_load_kwargs = config.get("tokenizer_load_kwargs", {})

	if loader not in ("transformers", "native_esm"):
		raise ValueError(f"Unknown loader type: {loader}. Supported loaders: 'transformers', 'native_esm'.")

	# Initialize model and tokenizer to None. They will be set based on the loader type.
	model = None
	tokenizer = None

	try:
		if loader == "transformers":
			# Load the model from pretrained weights.
			model = model_class.from_pretrained(hf_id, **load_kwargs)

			# Load tokenizer if a tokenizer class is specified in the configuration.
			if tokenizer_class:
				tokenizer = tokenizer_class.from_pretrained(hf_id, **tokenizer_load_kwargs)
				if tokenizer is None:
					raise RuntimeError(f"Failed to load tokenizer for {model_type} from {hf_id}.")

			# Cast to full-precision if no GPU is available:
			if device == "cpu":
				model.to(torch.float32)
		else:
			# Native ESM loader
			logger.error("Native ESM loader is currently not supported.")

		if model is None:
			raise RuntimeError(f"Failed to load model for {model_type}")
		
		if post_load_hook:
			post_load_hook(model)

		# Load the model onto the specified device and set it to evaluation mode.
		model.to(device).eval()

		logger.info(f"Model {model_type} successfully initialized on {device.type.upper()}.")
		return model, tokenizer
	except Exception as e:
		logger.error(f"Model {model_type} could not be loaded: {e}")
		sys.exit(1)


def compute_embeddings(model_type: str, df: pd.DataFrame, model: nn.Module, tokenizer: Optional[T5Tokenizer], only_last: bool):
	"""Compute embeddings for protein sequences.

	Model types supported:
	- prot_t5
	- prost_t5
	- prot_t5xxl
	
	Other models are not yet implemented.

	Parameters
	__________
	model_type (str):
		Model type to use for embedding.
	df (pd.DataFrame):
		DataFrame containing protein sequences and IDs. Must contain columns "processed_seq" and "protein_ID".
	model (nn.Module):
		Pretrained model to use for embedding.
	tokenizer (Optional[T5Tokenizer]):
		Tokenizer required for ProtT5/ProstT5 models.
	only_last (bool):
		Whether only the final representation embedding should be returned. If False, returns all hidden layers

	Returns
	_______
	dict:
		Mapping protein_ID -> embedding tensors.
	"""
	if df.empty:
		raise ValueError("DataFrame is empty. Please provide a DataFrame with protein sequences to embed.")
	if not "processed_seq" in df.columns:
		raise ValueError("DataFrame must contain a 'processed_seq' column with preprocessed protein sequences.")
	if not "protein_ID" in df.columns:
		raise ValueError("DataFrame must contain a 'protein_ID' column with unique identifiers for each protein sequence.")
	# Check for 'NA' protein_IDs and remove them from the DataFrame.
	bad_ids = df["protein_ID"].astype(str).str.upper().eq("NA")
	if bad_ids.any():
		# Remove rows with 'NA' protein_IDs to avoid issues during embedding.
		df = df[~bad_ids].copy()
		logger.warning(f"Removed {bad_ids.sum()} rows with 'NA' protein_IDs from DataFrame.")
		if df.empty:
			raise ValueError("DataFrame is empty after removing rows with 'NA' protein_IDs.")

	# Resulting dictionary to hold embeddings for each protein_ID.	
	embed_dict = {}

	# Run embedding. Currently, only "prot_t5", "prost_t5", "prot_t5xxl" are supported.
	if not model_type in ("prot_t5", "prost_t5", "prot_t5xxl"):
		raise NotImplementedError(f"Embedding with {model_type} is not implemented. Supported options: 'prot_t5', 'prost_t5', 'prot_t5xxl'.")
	else:
		if tokenizer is None:
			raise ValueError(f"Tokenizer is required for embedding with {model_type} but was not provided.")
		# Process sequences:
		for i in tqdm(range(0, len(df), 1), desc=f"Embedding {model_type}"):
			row = df.iloc[i]
			seq = row["processed_seq"]
			try:
				emb = embed_seq_T5(seq, model, tokenizer, only_last, model_type)
				embed_dict[row["protein_ID"]] = emb # 24 x (L, 1024) | 1 x (L, 1024)		   
				print(f"Embedding for {row['protein_ID']}:")
				print(f"{emb.shape if isinstance(emb, np.ndarray) else [e.shape for e in emb]}")
			except RuntimeError as e:
				if "seq_len" in row:
					logger.error(f"RunTime Error for sequence {i}, protein_ID: {row['protein_ID']}, length: {row['seq_len']}: {e}")
				else:
					logger.error(f"RunTime Error for sequence {i}: {e}")
				continue
	return embed_dict 


def embed_seq_T5(seq: str, model: T5EncoderModel, tokenizer: T5Tokenizer, only_last: bool, model_type: str):
	"""Embeds a single amino acid sequence using Prot_T5 model.

	Parameters
	__________
	seq (str):
		Protein sequence.
	model (T5EncoderModel):
		Prot_T5 model for embedding.
	tokenizer (T5Tokenizer):
		Prot_T5 tokenizer.
	only_last (bool):
		Whether only the final representation embedding should be returned. 
		Returns all hidden layers otherwise.
	model_type (str):
		Type of model to use for embedding. Options: "prot_t5", "prost_t5", "prot_t5xxl". Other models are not yet implemented.
							
	Returns
	_______
	last_layer_emb (np.ndarray):
		Array of final representation embedding.
	hidden_states (List[np.ndarray] | np.ndarray):
		List of arrays of shape (L, 1024) for each layer.
	"""
	tokenizer_kwargs = {
		"return_tensors": "pt",
		"truncation": False,
		"padding": True,
		"add_special_tokens": True,
	}
	inputs = tokenizer(seq, **tokenizer_kwargs).to(model.device)

	if only_last:
		with torch.no_grad():
			outputs = model(**inputs)

		if model_type in ("prot_t5", "prot_t5xxl"):
			# cut EOS token
			last_layer_emb = outputs.last_hidden_state.squeeze(0).cpu().numpy()[:-1, :]
		elif model_type == "prost_t5":
			# cut prefix and EOS tokens
			last_layer_emb = outputs.last_hidden_state.squeeze(0).cpu().numpy()[1:-1, :]

		return last_layer_emb

	else:
		# generate embeddings without tracking gradients (no training here)
		with torch.no_grad():
			# list of tensors, one per encoder layer, each shape (1, L, 1024)
			outputs = model(**inputs, output_hidden_states=True)

		all_hidden_states = outputs.hidden_states

		if model_type == "prot_t5":
			# list of np.ndarrays, each shape (L, 1024), cleaned
			hidden_states = [h.squeeze(0).cpu().numpy()[:-1, :] for h in all_hidden_states]
		elif model_type == "prost_t5":
			hidden_states = [h.squeeze(0).cpu().numpy()[1:-1, :] for h in all_hidden_states]

		return hidden_states
