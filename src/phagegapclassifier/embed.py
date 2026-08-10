"""
Protein sequence data and pLM model embedding utilities.

The `MODEL_CONFIGS` and `login_to_huggingface()` are adapted from
	https://github.com/tsenoner/plm_choice/blob/main/src/data_preparation/embeddings/embedding_generation.py
"""

from __future__ import annotations

import logging, sys, time, psutil, os
import pandas as pd
import numpy as np
from getpass import getpass
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from huggingface_hub import login as hf_login
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap


logger = logging.getLogger(__name__)

ModelConfig = Dict[str, Any]

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


def load_df(path_to_metadata_tsv: str) -> pd.DataFrame:
	"""Load metadata tsv containing protein information and protein sequences into a pandas DataFrame."""
	df = pd.read_csv(path_to_metadata_tsv, sep="\t")
	return df


def preprocess_df(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
	"""Preprocess protein sequence depending on which model type is chosen.

	Raises
	---------
	AssertionError 
		If df does not contain a column "protein_seq"
	"""
	assert "protein_seq" in df.columns, "df does not contain column 'protein_seq'."

	df["protein_seq"] = df["protein_seq"].str.upper()
	df["processed_seq"] = df["protein_seq"].str.replace("[UZOB]", "X", regex=True)

	# Prot T5 needs spaces between amino acids
	if model_type in ("prot_t5", "prot_t5xxl"):
		# only add spaces
		df["processed_seq"] = df["processed_seq"].apply(lambda aa: " ".join(list(aa)))
	elif model_type == "prost_t5":
		# add spaces and prefix for ProstT5
		df["processed_seq"] = df["processed_seq"].apply(lambda aa: "<AA2fold> " + " ".join(list(aa)))
	elif model_type == "glm2":
		# add prefix for gLM2
		df["processed_seq"] = df["processed_seq"].apply(lambda aa: "<+>" + "".join(list(aa)))

	return df


def load_model(checkpoint: str, device: str) -> Tuple[nn.Module, Optional[T5Tokenizer]]:
	"""Sets up the pLM model.

	Raises
	---------
	AssertionError 
		If checkpoint or loader is not supported.
	RunTimeError
		If model could not be loaded.
	"""
	if checkpoint not in MODEL_CONFIGS:
		raise ValueError(f"Unsupported model type: {checkpoint}. Supported types: {list(MODEL_CONFIGS.keys())}")
	# Load configuration for the specified checkpoint/model.
	config = MODEL_CONFIGS[checkpoint]
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

	# If the configuration requires login, attempt to log in to Hugging Face Hub.
	if config.get("requires_login", False):
		login_to_huggingface()
	try:
		if loader == "transformers":
			if checkpoint == "esm2_15b":
				logger.error("ESM2_15B model loading is currently not implemented.")
			elif checkpoint == "glm2":
				logger.error("gLM2 model loading is currently not implemented.")
			else:
				# Load the model from pretrained weights.
				model = model_class.from_pretrained(hf_id, **load_kwargs)

			# Load tokenizer if a tokenizer class is specified in the configuration.
			if tokenizer_class:
				tokenizer = tokenizer_class.from_pretrained(hf_id, **tokenizer_load_kwargs)
				if tokenizer is None:
					raise RuntimeError(f"Failed to load tokenizer for {checkpoint} from {hf_id}.")

			# Cast to full-precision if no GPU is available:
			if device == "cpu":
				model.to(torch.float32)
		else:
			# Native ESM loader
			logger.error("Native ESM loader is not fully implemented.")

		if model is None:
			raise RuntimeError(f"Failed to load model for {checkpoint}")
		
		if post_load_hook:
			post_load_hook(model)

		if checkpoint not in ("glm2", "esmc_6b"):
			model.to(device).eval()

		logger.info(f"Model {checkpoint} successfully initialized on {device.type.upper()}.")
		return model, tokenizer
	except Exception as e:
		logger.error(f"Model {checkpoint} could not be loaded: {e}")
		sys.exit(1)


def monitor_load_embed_model(checkpoint, device):
	process = psutil.Process(os.getpid())

	mem_before = process.memory_info().rss / 1024**2
	start = time.perf_counter()

	model, tokenizer = load_model(checkpoint, device)

	end = time.perf_counter()
	mem_after = process.memory_info().rss / 1024**2

	logger.info(f"Time: {end - start:.2f} s")
	logger.info(f"RAM:  {mem_after - mem_before:.2f} MB")

	return model, tokenizer


def compute_embeddings(checkpoint: str, df: pd.DataFrame, model: nn.Module, tokenizer: Optional[T5Tokenizer], only_last: bool):
	"""Compute embeddings for protein sequences using ProtT5.

	Params:
		checkpoint (str): Model type, specified in config.yaml
		df (pd.DataFrame): DataFrame containing protein sequences and IDs
		model (nn.Module): Pretrained model
		tokenizer (Optional[T5Tokenizer]): Tokenizer required for ProtT5, ProstT5, and gLM2
		only_last (bool): Whether only the final representation embedding should be returned.

	Returns:
		dict: Mapping protein_ID -> embedding tensors
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
	if checkpoint == "esm2_15b":
		raise NotImplementedError("esm2_15b embedding is currently not implemented.")
	elif checkpoint == "esmc_6b":
		raise NotImplementedError("esmc_6b embedding is currently not implemented.")
	else:
		if not checkpoint in ("prot_t5", "prost_t5", "prot_t5xxl"):
			raise NotImplementedError(f"Embedding with {checkpoint} is not implemented. Supported options: 'prot_t5', 'prost_t5', 'prot_t5xxl'.")
		if tokenizer is None:
			raise ValueError(f"Tokenizer is required for embedding with {checkpoint} but was not provided.")
		# Process sequences:
		for i in tqdm(range(0, len(df), 1), desc=f"Embedding {checkpoint}"):
			row = df.iloc[i]
			seq = row["processed_seq"]
			try:
				emb = embed_seq_T5(seq, model, tokenizer, only_last, checkpoint)
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


def embed_seq_T5(seq: str, model: T5EncoderModel, tokenizer: T5Tokenizer, only_last: bool, checkpoint: str):
	"""Embeds a single amino acid sequence using Prot_T5 model.

	Params:
		seq (str): Protein sequence
		model (T5EncoderModel): Prot_T5 model for embedding
		tokenizer (T5Tokenizer): Prot_T5 tokenizer
		only_last (bool): Whether only the final representation embedding should be returned. 
							Returns all hidden layers otherwise.
							
	Returns:
		last_layer_emb (np.ndarray): Array of final representation embedding
		hidden_states (List[np.ndarray] | np.ndarray): List of arrays of shape (L, 1024) for each layer
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

		if checkpoint in ("prot_t5", "prot_t5xxl"):
			# cut EOS token
			last_layer_emb = outputs.last_hidden_state.squeeze(0).cpu().numpy()[:-1, :]
		elif checkpoint == "prost_t5":
			# cut prefix and EOS tokens
			last_layer_emb = outputs.last_hidden_state.squeeze(0).cpu().numpy()[1:-1, :]

		return last_layer_emb

	else:
		# generate embeddings without tracking gradients (no training here)
		with torch.no_grad():
			# list of tensors, one per encoder layer, each shape (1, L, 1024)
			outputs = model(**inputs, output_hidden_states=True)

		all_hidden_states = outputs.hidden_states

		if checkpoint == "prot_t5":
			# list of np.ndarrays, each shape (L, 1024), cleaned
			hidden_states = [h.squeeze(0).cpu().numpy()[:-1, :] for h in all_hidden_states]
		elif checkpoint == "prost_t5":
			hidden_states = [h.squeeze(0).cpu().numpy()[1:-1, :] for h in all_hidden_states]

		return hidden_states


def login_to_huggingface(token_path_str: Optional[str] = None):
	"""Attempts to log in to Hugging Face Hub, optionally using a token from a specified path."""
	logger.info("Attempting to log in to Hugging Face Hub...")
	token = None
	token_file = None
	if token_path_str:
		token_file = Path(token_path_str)
	else:
		# Default token locations
		potential_paths = [
			Path.home() / ".cache" / "huggingface" / "token",
			Path.home() / ".huggingface" / "token",
		]
		for p_path in potential_paths:
			if p_path.is_file():
				token_file = p_path
				break

	if token_file and token_file.is_file():
		try:
			token = token_file.read_text().strip()
			logger.info(f"• Found Hugging Face token file at {token_file}")
		except Exception as e:
			logger.warning(f"• Could not read token from {token_file}: {e}")
			token = None

	try:
		if token:
			hf_login(token=token)
			logger.info("• Hugging Face login successful (used token from file).")
		else:
			logger.warning("• No token file specified or found in default locations. Attempting default login (e.g., cached session, env var).")
			hf_login()  # Attempts login using env variables or cached token.
			logger.info("• Hugging Face login successful or already authenticated (default method).")
	except Exception as login_exc:
		logger.warning(f"• Hugging Face login attempt failed: {login_exc}")
		logger.info("...Proceeding. This may fail if the model requires authentication for download.")
