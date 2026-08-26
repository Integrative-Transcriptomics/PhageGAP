"""
API endpoints for the `phagegapclassifier` Flask app.
"""

from __future__ import annotations

import logging
import hmac
import math
import pandas as pd
import numpy as np
from scipy.special import softmax
from flask import Blueprint, current_app, request, jsonify
from phagegapclassifier.embed import preprocess_df, compute_embeddings
from phagegapclassifier.pool import pool_embeddings
from phagegapclassifier.predict import run_prediction
from phagegapclassifier.utils import parse_sequence_data, prepare_for_pca


# Initialize logging configuration.
logger = logging.getLogger(__name__)
api_blueprint = Blueprint("api", __name__)


@api_blueprint.before_request
def require_api_token():
	"""Checks for a valid API token in the Authorization header of incoming requests.

	If the token is invalid, return a 401 Unauthorized response.
	"""
	api_token = _get_extension("api_token")
	if api_token is None:
		return None
	else:
		authorization = request.headers.get("Authorization", "")
		scheme, separator, supplied_token = authorization.partition(" ")

		valid = (
			separator
			and scheme.lower() == "bearer"
			and supplied_token
			and hmac.compare_digest(supplied_token, api_token)
		)

		if not valid:
			response = jsonify({"error": "Unauthorized request. Missing or invalid API token."})
			response.status_code = 401
			response.headers["WWW-Authenticate"] = "Bearer"
			return response

		return None


@api_blueprint.get("/active")
def get_active():
	"""`GET` request health-check endpoint to verify that the API is running."""
	# TODO: Adjust the response to include memory information.
	return "Ok", 200


@api_blueprint.get("/predict")
def get_prediction():
	"""`POST` request endpoint to predict functional classes for a given set of protein sequences.
	
	See :func:`phagegapclassifier.api._predict` for details.
	"""
	try:
		# Parse sequence data.
		sequence_text = request.get_data(as_text=True)

		# Run prediction and return results as JSON.
		result = _predict(sequence_text)
		return result.to_dict(orient="records"), 200
	except Exception as e:
		logger.exception(f"Prediction request failed: {str(e)}")
		return f"Prediction request failed: {str(e)}", 500


def _get_extension(key: str):
	"""Return a process-local app extensions for the specified key.
	
	See :func:`~phagegapclassifier.__init__` for the structure of the extension dictionary.

	Parameters
	__________
	key (str):
		The key of the extension to retrieve. Should be one of the keys in the app extensions dictionary.

	Returns
	_______
	Any:
		The value associated with the specified key in the app extensions dictionary or `None` if the key
		does not exist or an error occurs.
	"""
	try:
		return current_app.extensions["phagegap"][key]
	except Exception as e:
		logger.error(f"Failed to access PhageGAP extension '{key}': {str(e)}")
		raise RuntimeError(f"Failed to access PhageGAP extension '{key}': {str(e)}") from e


def _predict(sequence_text: str) -> pd.DataFrame:
	"""Main prediction function of the PhageGAP classifier.
	
	Parameters
	__________
	sequence_text (str):
		A string containing protein sequences in FASTA format.

	Returns
	_______
	pd.DataFrame:
		A DataFrame containing predictions, probabilities, and projection information for each protein.
	"""
	# Load application extensions needed for prediction.
	try:
		config = _get_extension("config")
		plm = _get_extension("plm")
		plm_model_type = _get_extension("tokenizer")
		plm = _get_extension("plm")
		tokenizer = _get_extension("tokenizer")
		classifier = _get_extension("classifier")
		label_map = _get_extension("label_map")
		pca = _get_extension("pca")
		tsne = _get_extension("tsne")
		kdtree = _get_extension("kdtree")
		knn = _get_extension("knn")
	except Exception as e:
		logger.exception(f"Failed to load required extensions for prediction: {str(e)}")
		raise RuntimeError(f"Failed to load required extensions for prediction: {str(e)}") from e

	# Load configuration parameters for embedding pooling.
	# TODO: Consider informing the user if these parameters are missing or invalid in the configuration.
	plm_model_type = config.get("embed", {}).get("model_type", "prot_t5")
	pool_layers = config.get("embed", {}).get("pool_layers", [])
	pool_strategy = config.get("embed", {}).get("pool_strategy", "mean")

	# Parse and preprocess sequence data.
	try:
		df = parse_sequence_data(sequence_text)
	except Exception as e:
		logger.exception(f"Failed to parse sequence data: {str(e)}")
		raise RuntimeError(f"Failed to parse sequence data: {str(e)}")
	if df.empty:
		logger.error("Request contains no valid FASTA records.")
		raise RuntimeError("Request contains no valid FASTA records.")
	preprocessed_df = preprocess_df(df, plm_model_type)

	# Compute pLM embeddings.
	embed_dict = compute_embeddings(
		plm_model_type,
		preprocessed_df,
		plm,
		tokenizer,
		True,
	)

	# Pool embeddings according to the specified strategy and layers.
	embed_dict_pooled = pool_embeddings(
		embed_dict,
		plm_model_type,
		layers=pool_layers,
		strategy=pool_strategy,
	)

	# Predict protein classes.
	predictions_df, _ = run_prediction(
		embed_dict,
		preprocessed_df,
		classifier,
		label_map,
	)

	# Drop the processed sequence column if it exists.
	predictions_df.drop(
		columns=["processed_seq"],
		inplace=True,
		errors="ignore",
	)

	# Project the embedded proteins into PCA space and search nearest neighbors.
	embed_matrix, embed_protein_ids = prepare_for_pca(embed_dict_pooled)
	embed_pcs = pca["model"].transform(embed_matrix)
	nn_pca_distances, nn_indices = kdtree.query(embed_pcs, k=3)
	nn_pca_distances = nn_pca_distances.tolist() # (m, knn) 2D list.
	nn_indices = nn_indices.tolist() # (m, knn) 2D list.
	
	# Extract nearest neighbor information.
	for indices_list, pca_distances_list in zip(nn_indices, nn_pca_distances):
		# Extract the t-SNE coordinates of the nearest neighbors.
		neighbor_tsne_coords = [ tsne["coords"][i] for i in indices_list ]

	"""
	Needs:
	- ...
	"""
	
	projection_df = pd.DataFrame({
		"protein_ID": embed_protein_ids,
		"nearest_neighbor_information": [ pca["ids"][_[0]] for _ in nn_indices ],
	})

	# Merge predictions with projection data and return as JSON.
	result = pd.merge(predictions_df, projection_df, on="protein_ID", how="left")
	return result
