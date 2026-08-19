"""
Utility functions of the `phagegapclassifier` package.
"""

from __future__ import annotations

import random
import torch
import logging
import re
import numpy as np
import pandas as pd
from io import StringIO
from Bio import SeqIO
from transformers import set_seed


# Initialize logging configuration.
logger = logging.getLogger(__name__)


def set_determinism(seed: int) -> None:
	"""Set seeds for python, numpy, transformers and torch.

	Parameters
	__________
	seed (int):
		The seed value to set for all random number generators.

	Raises
	______
	AssertionError
		If seed is negative.
	"""
	assert seed >= 0, "Seed must be non-negative."
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	set_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True, warn_only=True)


def extract_model_number(path: str) -> int:
	"""Extracts number of model used to make predictions.

	Parameters
	__________
	path (str):
		Path to the model pickle file.

	Returns
	_______
	int:
		The extracted model number, or -1 if not found.
	"""
	match = re.search(r"model(\d+)\_final.pt$", path)
	if match:
		num = int(match.group(1))
	return num


def parse_sequence_data(sequence_data: str) -> pd.DataFrame:
	"""Parses sequence data in FASTA format and returns a DataFrame with protein IDs, descriptions, and sequences.

	Parameters
	__________
	sequence_data (str):
		A string containing sequence data in FASTA format.
	
	Returns
	_______
	pd.DataFrame:
		A data frame with columns `protein_ID`, `description`, and `protein_seq`.
	"""
	records = [
		{
			"protein_ID": str(record.id),
			"description": str(record.description),
			"protein_seq": str(record.seq),
		}
		for record in SeqIO.parse(
			StringIO(sequence_data),
			"fasta",
		)
	]
	return pd.DataFrame.from_records(
		records,
		columns=[
			"protein_ID",
			"description",
			"protein_seq",
		],
	)


def prepare_for_pca(embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
	"""Prepares an embedding dictionary for PCA transformation.

	Parameters
	__________
	embeddings (Dict[str, np.ndarray]):
		A dictionary where keys are protein IDs and values are their corresponding embeddings.

	Returns
	_______
	tuple[np.ndarray, list[str]]:
		A tuple containing:
		- X: A 2D numpy array of shape (n_proteins, embed_dim) containing the embeddings.
		- y: A list of protein IDs corresponding to the rows in X.
	"""
	X, y = [], []
	for key, value in embeddings.items():
		X.append(value)
		y.append(key)
	X = np.stack(X) # shape (n_proteins, embed_dim)
	return X, y
