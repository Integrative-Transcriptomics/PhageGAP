"""
Pooling utilities for pLM protein embeddings.
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AttentionPool(nn.Module):
	"""
	Attention pooling for protein embeddings. This layer computes attention scores for each token in the sequence and uses these scores to compute a weighted sum of the token embeddings, resulting in a single pooled embedding vector.
	"""
	def __init__(self, emb_dim):
		out_dim = emb_dim
		super().__init__()
		self.attn = nn.Linear(emb_dim, 1)
		self.fc = nn.Linear(emb_dim, out_dim)
	
	def forward(self, x): # x: (L, embed_dim)
		scores = self.attn(x) # (L, 1)
		weights = torch.softmax(scores, 0) # (L, 1)
		pooled = (weights * x).sum(dim=0) # (embed_dim,)
		return self.fc(pooled) # (out_dim,)


def filter_nan_embeddings(
		embeddings: Dict[str, np.ndarray | List[np.ndarray]],
		metadata_df: pd.DataFrame
		) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
	"""
	Remove embeddings that contain only zeros (as for ESM-3 not all proteins could be embedded).
	"""
	
	def is_all_nan(embedding):
		"""
		Helper function to check if an embedding is all NaN values. Works for both single-layer and multi-layer embeddings.
		"""
		if isinstance(embedding, list):
			return all(np.all(np.isnan(e)) for e in embedding)
		else:
			return np.all(np.isnan(embedding))

	# Identify zero vector protein IDs.
	nan_vector_proteins = [protein_id for protein_id, embedding in embeddings.items() if is_all_nan(embedding)]
	# Remove them from embed_dict.
	filtered_embed_dict = {protein_id: embedding for protein_id, embedding in embeddings.items() if protein_id not in nan_vector_proteins}
	# Remove corresponding rows in the metadata_df.
	filtered_metadata_df = metadata_df[~metadata_df['protein_ID'].isin(nan_vector_proteins)]

	return filtered_embed_dict, filtered_metadata_df


def pool_embeddings(embeddings: Dict[str, List[np.ndarray] | np.ndarray], checkpoint: str, layers: List[int], strategy: str) -> Dict[str, np.ndarray]:
	"""
	Pools specified hidden layers of protein sequence embeddings using a given strategy (e.g. mean/max pooling). 

	Params
	----------
	embeddings (Dict[str, List[np.ndarray] | np.ndarray]): Dictionary mapping protein_ID -> (List of) embedding(s), each shape (L, embed_dim) for each layer
	checkpoint (str): The pLM used for creating protein embeddings
	layers (List[int]): The hidden layers specified to be used for pooling. Rest is ignored. If no layers are given, uses last hidden layer.
	strategy (str): Pooling strategy 

	Returns
	-----------
	Dict[str, np.ndarray]: Dictionary mapping protein_ID -> pooled embedding vector (np.ndarray)
	"""
	assert checkpoint in ("esm2_150m", "esm2_650m", "esm2_3b", "prot_t5", "prot_t5xxl", "esmc_300m", "esmc_600m", "esmc_6b", "esm3_open", "prost_t5", "glm2_650m"), f"Unsupported checkpoint: {checkpoint}"
	assert strategy in ("mean", "max", "FC"), f"Unsupported pooling strategy: {strategy}"
	
	if checkpoint in ("prot_t5", "prot_t5xxl", "prost_t5"):
		max_layer = 23
		embed_dim = 1024
	elif checkpoint == "esmc_300m":
		max_layer = 29
		embed_dim = 960
	elif checkpoint == "esmc_600m":
		max_layer = 35
		embed_dim = 1152
	elif checkpoint == "esmc_6b":
		max_layer = 79
		embed_dim = 2560
	elif checkpoint == "esm3_open":
		max_layer = 47
		embed_dim = 1536
	elif checkpoint == "glm2_650m":
		max_layer = 29
		embed_dim = 1280
	elif checkpoint == "esm2_150m":
		max_layer = 29
		embed_dim = 640
	elif checkpoint == "esm2_650m":
		max_layer = 32
		embed_dim = 1280
	elif checkpoint == "esm2_3b":
		max_layer = 35
		embed_dim = 2560

	pooled_embeddings = {}

	# Set up attention pooling for fully connected layer strategy.
	pooler = AttentionPool(emb_dim=embed_dim)
	pooler.eval()
	reported = False

	for pid, all_layers in tqdm(embeddings.items(), total=len(embeddings), desc="Pooling embeddings..."):
		# Single-layer case:
		if isinstance(all_layers, np.ndarray) and (reported == False):
			max_layer = 0
			all_layers_list = [all_layers]
			logger.warning(f"Using only available layer for pooling, ignoring provided layers list.")
			reported = True
		elif isinstance(all_layers, np.ndarray) and (reported == True):
			max_layer = 0
			all_layers_list = [all_layers]
		else:
			# Multi-layer case:
			all_layers_list = all_layers
			for l in layers:
				assert 0 <= l <= max_layer, f"Layer {l} out of range for {checkpoint} (max={max_layer})"

		# Take the last (or only) hidden layer, if the layer list is empty.
		if len(layers) == 0:
			layers = [max_layer]		 
		extracted_layers = [all_layers_list[i] for i in layers] # Each of shape (L, embed_dim)
		

		if strategy == "mean":
			# 1) Pool over sequence length.
			layer_vectors = [np.mean(layer, axis=0) for layer in extracted_layers]
			# 2) Pool over layers.
			final_vector = np.mean(np.stack(layer_vectors), axis=0)
		elif strategy == "max":
			# 1) Pool over sequence length.
			layer_vectors = [np.max(layer, axis=0) for layer in extracted_layers]
			# 2) Pool over layers.
			final_vector = np.max(np.stack(layer_vectors), axis=0)
		else:
			with torch.no_grad():
				# 1) Pool over sequence length.
				layer_vectors = []
				for layer in extracted_layers:
					layer = torch.from_numpy(layer).float()
					layer_vectors.append(pooler(layer))

				# 2) Pool over layers.
				final_vector = torch.stack(layer_vectors).mean(dim=0).detach().cpu().numpy()

		pooled_embeddings[pid] = final_vector

	return pooled_embeddings
