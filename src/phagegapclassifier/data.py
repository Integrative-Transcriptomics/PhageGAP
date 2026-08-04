"""
Data parsing utilities.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

from Bio import SeqIO
from io import StringIO


def parse_sequence_data(sequence_data: str) -> pd.DataFrame:
	"""
	Parse FASTA-formatted protein sequences.

	Parameters
	----------
	sequence_data (str):
		A string containing sequence data in FASTA format.
	
	Returns
	-------
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


def extract_model_number(path: str) -> int:
	"""
	Extracts number of model used to make predictions.

	Parameters
	----------
	path (str):
		Path to the model pickle file.

	Returns
	-------
	int:
		The extracted model number, or -1 if not found.
	"""
	match = re.search(r"model(\d+)\_final.pt$", path)
	if match:
		num = int(match.group(1))
	return num


def prepare_for_pca(embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
	"""
	Prepares an embedding dictionary for PCA transformation.

	Parameters:
	-----------
	embeddings (Dict[str, np.ndarray]):
		A dictionary where keys are protein IDs and values are their corresponding embeddings.
	"""
	X, y = [], []
	for key, value in embeddings.items():
		X.append(value)
		y.append(key)
	X = np.stack(X) # shape (n_proteins, embed_dim)
	return X, y
