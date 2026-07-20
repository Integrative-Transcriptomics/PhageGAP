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
    Parses sequence data from a FASTA format string into a pandas data frame.
    """
    return pd.DataFrame.from_records(
        [
            {
                "protein_ID": record.id,
                "description": record.description,
                "protein_seq": str(record.seq)
            } for record in SeqIO.parse(StringIO(sequence_data), "fasta")
        ]
    )


def extract_model_number(path: str):
    """
    Extracts number of model used to make predictions.
    """
    match = re.search(r"model(\d+)\_final.pt$", path)
    if match:
        num = int(match.group(1))
    return num


def prepare_for_pca(embeddings):
    """
    Prepares embedding dictionary for PCA transformation. Returns X and y.
    """
    X, y = [], []
    for key, value in embeddings.items():
        X.append(value)
        y.append(key)
    X = np.stack(X) # shape (n_proteins, embed_dim)
    return X, y
