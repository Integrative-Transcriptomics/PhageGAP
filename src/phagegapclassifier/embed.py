"""
Protein sequence data and pLM model embedding utilities.

The `MODEL_CONFIGS` and `login_to_huggingface()` are adapted from
	https://github.com/tsenoner/plm_choice/blob/main/src/data_preparation/embeddings/embedding_generation.py

TODO: Remove unused models for release.
"""

from __future__ import annotations

import logging, sys, time, psutil, os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from getpass import getpass
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import esm
from esm.models.esmc import ESMC
from esm.models.esm3 import ESM3
from esm.sdk import client
from esm.sdk.api import ESMProtein, LogitsConfig, SamplingConfig, ESM3InferenceClient
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, AutoModel, AutoTokenizer, EsmModel
from huggingface_hub import login as hf_login
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap


logger = logging.getLogger(__name__)

ModelConfig = Dict[str, Any]

MODEL_CONFIGS: Dict[str, ModelConfig] = {
	"glm2_650m": {
		"hf_id": "tattabio/gLM2_650M", #'tattabio/gLM2_650M_embed' | "tattabio/gLM2_650M"
		"loader": "transformers",
		"model_class": AutoModel,
		"tokenizer_class": AutoTokenizer,
		"requires_explicit_login": False,
		"load_kwargs": {"torch_dtype": torch.bfloat16, "trust_remote_code": True},
		"tokenizer_load_kwargs": {"trust_remote_code": True},
	},
	"esm2_150m": {
		"hf_id": "facebook/esm2_t30_150M_UR50D",
		"loader": "transformers",
		"model_class": EsmModel,
		"tokenizer_class": AutoTokenizer,
	},
	"esm2_650m": {
		"hf_id": "facebook/esm2_t33_650M_UR50D",
		"loader": "transformers",
		"model_class": EsmModel,
		"tokenizer_class": AutoTokenizer,
	},
	"esm2_3b": {
		"hf_id": "facebook/esm2_t36_3B_UR50D",
		"loader": "transformers",
		"model_class": EsmModel,
		"tokenizer_class": AutoTokenizer,
	},
	"esm2_15b": {
		"hf_id": "facebook/esm2_t48_15B_UR50D",
		"loader": "transformers",
		"model_class": EsmModel,
		"tokenizer_class": AutoTokenizer,
	},
	"esmc_300m": {
		"hf_id": "esmc_300m",
		"loader": "native_esm",
		"model_class": ESMC,
		"tokenizer_class": None,
		"requires_explicit_login": True
	},
	"esmc_600m": {
		"hf_id": "esmc_600m",
		"loader": "native_esm",
		"model_class": ESMC,
		"tokenizer_class": None,
		"requires_explicit_login": True,
	},
	"esmc_6b": {
		"hf_id": "esmc-6b-2024-12",
		"loader": "native_esm",
		"model_class": ESMC,
		"tokenizer_class": None,
		"requires_explicit_login": True
	},
	"esm3_open": {
		"hf_id": "esm3-open",
		"loader": "native_esm",
		"model_class": ESM3,
		"tokenizer_class": None,
		"requires_explicit_login": True,
	},
	"esm3_7b": {
		"hf_id": "esm3-medium-2024-03",
		"loader": "native_esm",
		"model_class": ESM3,
		"tokenizer_class": None,
		"requires_explicit_login": True,
	},
	"esm3_98b": {
		"hf_id": "esm3-large-2024-03",
		"loader": "native_esm",
		"model_class": ESM3,
		"tokenizer_class": None,
		"requires_explicit_login": True,
	},
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

ESMC_EMBEDDING_CONFIG = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
ESM3_EMBEDDING_CONFIG = SamplingConfig(return_per_residue_embeddings=True)


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
	assert "protein_seq" in df.columns, "df does not contain column 'protein_seq'"

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
	assert checkpoint in MODEL_CONFIGS.keys(), "unsupported model type"
	
	config = MODEL_CONFIGS[checkpoint]

	hf_id = config["hf_id"]
	loader = config["loader"]
	model_class = config["model_class"]
	tokenizer_class = config["tokenizer_class"]
	load_kwargs = config.get("load_kwargs", {})
	post_load_hook = config.get("post_load_hook")
	tokenizer_load_kwargs = config.get("tokenizer_load_kwargs", {})

	assert loader in ("transformers", "native_esm"), f"Unknown loader type: {loader}"

	model = None
	tokenizer = None

	if config.get("requires_login", False):
		login_to_huggingface()

	try:
		if loader == "transformers":
			if checkpoint == "esm2_15b":
				# init the distributed world with world_size 1
				url = "tcp://localhost:23456"
				torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

				# download model data from the hub
				model_name = "esm2_t48_15B_UR50D"
				model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)

				# initialize the model with FSDP wrapper
				fsdp_params = dict(
					mixed_precision=True,
					flatten_parameters=True,
					state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
					cpu_offload=True,  # enable cpu offloading
				)

				with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
					model, vocab = esm.pretrained.load_model_and_alphabet_core(
						model_name, model_data, regression_data
					)
					batch_converter = vocab.get_batch_converter()
					model.eval()

					# Wrap each layer in FSDP separately
					for name, child in model.named_children():
						if name == "layers":
							for layer_name, layer in child.named_children():
								wrapped_layer = wrap(layer)
								setattr(child, layer_name, wrapped_layer)
					model = wrap(model)

				return model, batch_converter

			elif checkpoint == "glm2":
				model = model_class.from_pretrained(hf_id, **load_kwargs).cuda()

			else:
				model = model_class.from_pretrained(hf_id, **load_kwargs)

			if tokenizer_class:
				tokenizer = tokenizer_class.from_pretrained(hf_id, **tokenizer_load_kwargs)

			# cast to full-precision if no GPU is available
			if device == "cpu":
				model.to(torch.float32)

		else:
			if checkpoint in ("esmc_6b", "esm3_98b"):
				token = getpass("Token from Forge: ")
				model = client(model=hf_id, url="https://forge.evolutionaryscale.ai", token=token)

			else:
				model = model_class.from_pretrained(hf_id)

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


def embed_seq_glm2(seq: str, model: AutoModel, tokenizer: AutoTokenizer, only_last: bool):
	"""Embeds a single amino acid sequence using gLM2
	
	Params:
		seq (str): Protein sequence
		model (AutoModel): gLM2 model for embedding
		tokenizer (AutoTokenizer): tokenizer for sequence
		only_last (bool): Whether only the final representation embedding should be returned. 
							Returns all hidden layers otherwise.
							
	Returns:
		last_layer_emb (np.ndarray): Array of final representation embedding
		hidden_states (List[np.ndarray] | np.ndarray): List of arrays of shape (L, 1280) for each layer
	"""
	device = next(model.parameters()).device
	# tokenize sequence
	protein_tensor = tokenizer([seq], return_tensors='pt')
	# remove unsupported keys
	protein_tensor.pop("token_type_ids", None)
	protein_tensor = {k: v.to(device) for k, v in protein_tensor.items()}

	if only_last:
		# generate embeddings / logits 
		with torch.no_grad():
			output = model(**protein_tensor, output_hidden_states=True).last_hidden_state
		# get final model embedding (final representation layer)
		last_layer_emb = output.squeeze(0).float().cpu().numpy()
		return last_layer_emb

	else:
		# get list of per-layer hidden states
		with torch.no_grad():
			output = model(**protein_tensor, output_hidden_states=True)
	
		hidden_states = []
		for h in output.hidden_states:
			h = h.squeeze(0).to(dtype=torch.float32).cpu()
			hidden_states.append(h.numpy())

		return hidden_states


def embed_seq_esmc(seq: str, model: ESM3InferenceClient, only_last: bool, checkpoint: str):
	"""Embeds a single amino acid sequence using ESM-C model.

	Params:
		seq (str): Protein sequence
		model (ESM3InferenceClient): ESM-C model for embedding
		only_last (bool): Whether only the final representation embedding should be returned. 
							Returns all hidden layers otherwise.
							
	Returns:
		last_layer_emb (np.ndarray): Array of final representation embedding
		hidden_states (List[np.ndarray] | np.ndarray): List of arrays of shape (L, 960) for each layer
	"""
	protein = ESMProtein(sequence=seq)
	protein_tensor = model.encode(protein)
	
	# generate embeddings / logits 
	if checkpoint == "esmc_6b":
		output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
	else:
		output = model.logits(protein_tensor.to(model.device), ESMC_EMBEDDING_CONFIG)

	if only_last:
		# get final model embedding (final representation layer)
		if checkpoint == "esmc_6b":
			last_layer_emb = output.embeddings.squeeze(0).cpu().to(torch.float32)[1:-1, :].numpy()
		else:
			last_layer_emb = output.embeddings.squeeze(0).cpu()[1:-1, :].numpy()
		return last_layer_emb

	else:
		# get list of per-layer hidden states, remove BOS/EOS tokens
		hidden_states = []
		for h in output.hidden_states:
			h = h.squeeze(0).to(dtype=torch.float32).cpu()
			hidden_states.append(h[1:-1, :].numpy())

		return hidden_states


def embed_seq_esmc6b(seq: str, model: ESM3InferenceClient):
	protein = ESMProtein(sequence=seq)
	protein_tensor = model.encode(protein)
	
	# generate embeddings / logits 
	output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
	last_layer_emb = output.embeddings.squeeze(0).cpu().to(torch.float32)[1:-1, :].numpy()

	return last_layer_emb


def embed_seq_esm3(seq: str, model: ESM3InferenceClient, only_last: bool):
	"""Embeds a single amino acid sequence using ESM-C model.

	Params:
		seq (str): Protein sequence
		model (ESM3InferenceClient): ESM-C model for embedding
		only_last (bool): Whether only the final representation embedding should be returned. 
							Returns all hidden layers otherwise.
							
	Returns:
		last_layer_emb (np.ndarray): Array of final representation embedding
		hidden_states (List[np.ndarray] | np.ndarray): List of arrays of shape (L, 1536) for each layer
	"""
	protein = ESMProtein(sequence=seq)
	protein_tensor = model.encode(protein)
	
	# generate embeddings / logits 
	if only_last:
		with torch.no_grad():
			output = model.forward_and_sample(protein_tensor.to(model.device), ESM3_EMBEDDING_CONFIG)
		
		# get final model embedding (final representation layer)
		last_layer_emb = output.per_residue_embedding.squeeze(0).cpu()[1:-1, :].numpy()
		
		return last_layer_emb

	else:
		# get list of per-layer hidden states, remove BOS/EOS tokens
		with torch.no_grad():
			output = model(protein_tensor.to(model.device), return_embeddings=True, return_hidden_states=True)
		
		hidden_states = []
		for h in output.hidden_states:
			h = h.squeeze(0).to(dtype=torch.float32).cpu()
			hidden_states.append(h[1:-1, :].numpy())

		return hidden_states


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


def embed_seq_esm2(seq: str, model: EsmModel, tokenizer: AutoTokenizer, only_last: bool):
	"""Embeds a single amino acid sequence using ESM-2 model.

	Params:
		seq (str): Protein sequence
		model (EsmModel): ESM-2 model for embedding
		tokenizer (AutoTokenizer): ESM-2 tokenizer
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

		# remove cls and eos tokens
		last_layer_emb = outputs.last_hidden_state.squeeze(0).cpu().numpy()[1:-1, :]

		return last_layer_emb

	else:
		# generate embeddings without tracking gradients (no training here)
		with torch.no_grad():
			# list of tensors, one per encoder layer, each shape (1, L, 1024)
			outputs = model(**inputs, output_hidden_states=True)

		all_hidden_states = outputs.hidden_states

		hidden_states = [h.squeeze(0).cpu().numpy()[1:-1, :] for h in all_hidden_states]

		return hidden_states


def compute_embeddings(df: pd.DataFrame, model: nn.Module):
	embed_dict = {}
	with ThreadPoolExecutor() as executor:
		futures = {executor.submit(embed_seq_esmc6b, seq, model): (protein_id, seq) for protein_id, seq in zip(df["protein_ID"], df["processed_seq"])}

		for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding esmc_6b"):
			protein_id, seq = futures[future]
			try:
				embed_dict[protein_id] = future.result()
			except Exception as e:
				logger.error(f"RunTime Error for protein_ID: {protein_id}, length: {len(seq)}: {e}")
				embed_dict[protein_id] = np.full((len(seq), 2560), np.nan)

	return embed_dict


def compute_embeddings(checkpoint: str, df: pd.DataFrame, model: nn.Module, tokenizer: Optional[T5Tokenizer], only_last: bool):
	"""Compute embeddings for protein sequences using ProtT5 or ESM-C.

	Params:
		checkpoint (str): Model type, specified in config.yaml
		df (pd.DataFrame): DataFrame containing protein sequences and IDs
		model (nn.Module): Pretrained model
		tokenizer (Optional[T5Tokenizer]): Tokenizer required for ProtT5, ProstT5, and gLM2
		only_last (bool): Whether only the final representation embedding should be returned.

	Returns:
		dict: Mapping protein_ID -> embedding tensors
	"""

	assert "processed_seq" in df.columns, "no preprocessed protein sequences in df available."
	assert "protein_ID" in df.columns, "Column 'protein_ID' is missing in DataFrame."
	bad_ids = df["protein_ID"].astype(str).str.upper().eq("NA")
	assert not bad_ids.any(), "Error: 'protein_ID' column contains 'NA' values."
	assert only_last in (True, False)
	
	embed_dict = {}

	if checkpoint == "esm2_15b":
		assert only_last, "esm2_15b only supports only_last=True"

		sequences = list(zip(df["protein_ID"], df["processed_seq"]))
		batch_size = 1

		for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding esm2_15b"):
			batch = sequences[i:i + batch_size]
			batch_labels, batch_strs, batch_tokens = tokenizer(batch)

			#batch_tokens = batch_tokens.cuda()

			with torch.no_grad():
				outputs = model(batch_tokens, repr_layers=[48], return_contacts=False)

			token_representations = outputs["representations"][48]

			for j, (protein_id, seq) in enumerate(batch):
				# remove CLS + EOS tokens
				emb = token_representations[j, 1:len(seq)+1].cpu().numpy()
				embed_dict[protein_id] = emb

	elif checkpoint == "esmc_6b":
		if not only_last:
			raise NotImplementedError("Only last layer embedding is supported for esmc_6b due to memory constraints.")
		
		with ThreadPoolExecutor() as executor:
			futures = {executor.submit(embed_seq_esmc6b, seq, model): (protein_id, seq) for protein_id, seq in zip(df["protein_ID"], df["processed_seq"])}

			for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding esmc_6b"):
				protein_id, seq = futures[future]
				try:
					embed_dict[protein_id] = future.result()
				except Exception as e:
					logger.error(f"RunTime Error for protein_ID: {protein_id}, length: {len(seq)}: {e}")
					embed_dict[protein_id] = np.full((len(seq), 2560), np.nan)

		return embed_dict

	else:
		# process sequences
		for i in tqdm(range(0, len(df), 1), desc=f"Embedding {checkpoint}"):
			row = df.iloc[i]
			seq = row["processed_seq"]

			try:
				if checkpoint in ("prot_t5", "prost_t5", "prot_t5xxl"):
					assert tokenizer is not None, f"{checkpoint}: tokenizer not instantiated."

					emb = embed_seq_T5(seq, model, tokenizer, only_last, checkpoint)
					embed_dict[row["protein_ID"]] = emb # 24 x (L, 1024) | 1 x (L, 1024)

				elif checkpoint == "glm2_650m":
					assert tokenizer is not None, f"{checkpoint}: tokenizer not instantiated."
					emb = embed_seq_glm2(seq, model, tokenizer, only_last)
					embed_dict[row["protein_ID"]] = emb # 30 x (L, 1280) | 1 x (L, 1280)

				elif checkpoint in ("esmc_300m", "esmc_600m", "esmc_6b"):
					emb = embed_seq_esmc(seq, model, only_last, checkpoint)
					embed_dict[row["protein_ID"]] = emb 

				elif checkpoint in ("esm2_150m", "esm2_650m", "esm2_3b", "esm2_15b"):
					assert tokenizer is not None, f"{checkpoint}: tokenizer not instantiated."
					emb = embed_seq_esm2(seq, model, tokenizer, only_last)
					embed_dict[row["protein_ID"]] = emb	 

				else:
					emb = embed_seq_esm3(seq, model, only_last)
					embed_dict[row["protein_ID"]] = emb # 48 x (L, 1536) | 1 x (L, 1536)			   

				print(f"Embedding for {row['protein_ID']}:")
				print(f"{emb.shape if isinstance(emb, np.ndarray) else [e.shape for e in emb]}")

			except RuntimeError as e:
				if "seq_len" in row:
					logger.error(f"RunTime Error for sequence {i}, protein_ID: {row['protein_ID']}, length: {row['seq_len']}: {e}")
				else:
					logger.error(f"RunTime Error for sequence {i}")
				if checkpoint in ("prot_t5", "prost_t5"):
					if only_last:
						embed_dict[row["protein_ID"]] = np.full((len(seq), 1024), np.nan)
					else:
						embed_dict[row["protein_ID"]] = [np.full((len(seq), 1024), np.nan) for _ in range(24)] # assume 1024-dim. embedding, 24 layers
				
				elif checkpoint == "glm2_650m":
					if only_last:
						embed_dict[row["protein_ID"]] = np.full((len(seq), 1280), np.nan)
					else:
						embed_dict[row["protein_ID"]] = [np.full((len(seq), 1280), np.nan) for _ in range(30)] # assume 1280-dim. embedding, 30 layers
				
				elif checkpoint == "esmc_300m":
					if only_last:
						embed_dict[row["protein_ID"]] = np.full((len(seq), 960), np.nan)
					else:
						embed_dict[row["protein_ID"]] = [np.full((len(seq), 960), np.nan) for _ in range(30)] # assume 960-dim. embedding, 30 layers

				elif checkpoint == "esmc_600m":
					if only_last:
						embed_dict[row["protein_ID"]] = np.full((len(seq), 1152), np.nan)
					else:
						embed_dict[row["protein_ID"]] = [np.full((len(seq), 1152), np.nan) for _ in range(36)] # assume 1152-dim. embedding, 36 layers

				elif checkpoint == "esmc_6b":
					if only_last:
						embed_dict[row["protein_ID"]] = np.full((len(seq), 2560), np.nan)
					else:
						embed_dict[row["protein_ID"]] = [np.full((len(seq), 2560), np.nan) for _ in range(80)] # assume 2560-dim. embedding, 80 layers			   

				elif checkpoint == "esm2_150m":
					if only_last:
						embed_dict[row["protein_ID"]] = np.full((len(seq), 640), np.nan)
					else:
						embed_dict[row["protein_ID"]] = [np.full((len(seq), 640), np.nan) for _ in range(30)] # assume 640-dim. embedding, 30 layers

				elif checkpoint == "esm2_650m":
					if only_last:
						embed_dict[row["protein_ID"]] = np.full((len(seq), 1280), np.nan)
					else:
						embed_dict[row["protein_ID"]] = [np.full((len(seq), 1280), np.nan) for _ in range(33)] # assume 1280-dim. embedding, 33 layers

				elif checkpoint == "esm2_3b":
					if only_last:
						embed_dict[row["protein_ID"]] = np.full((len(seq), 2560), np.nan)
					else:
						embed_dict[row["protein_ID"]] = [np.full((len(seq), 2560), np.nan) for _ in range(36)] # assume 2560-dim. embedding, 36 layers

				elif checkpoint == "esm2_15b":
					if only_last:
						embed_dict[row["protein_ID"]] = np.full((len(seq), 5120), np.nan)
					else:
						embed_dict[row["protein_ID"]] = [np.full((len(seq), 5120), np.nan) for _ in range(48)] # assume 5120-dim. embedding, 48 layers

				else: # esm3
					if only_last:
						embed_dict[row["protein_ID"]] = np.full((len(seq), 1536), np.nan)
					else:
						embed_dict[row["protein_ID"]] = [np.full((len(seq), 1536), np.nan) for _ in range(48)] # assume 1536-dim. embedding, 48 layers
				continue

	return embed_dict 


def login_to_huggingface(token_path_str: Optional[str] = None):
	"""Attempts to log in to Hugging Face Hub, optionally using a token from a specified path."""
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
			print(f"Found Hugging Face token file at: {token_file}")
		except Exception as e:
			print(f"Could not read token from {token_file}: {e}")
			token = None

	try:
		if token:
			hf_login(token=token)
			print("Hugging Face login successful (used token from file).")
		else:
			print("No token file specified or found in default locations. Attempting default login (e.g., cached session, env var).")
			hf_login()  # Attempts login using env variables or cached token
			print("Hugging Face login successful or already authenticated (default method).")
	except Exception as login_exc:
		print(f"Hugging Face login attempt failed: {login_exc}")
		print("   Proceeding. This may fail if the model requires authentication for download.")
