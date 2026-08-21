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
def is_active():
	"""`GET` request health-check endpoint to verify that the API is running."""
	return 1, 200


@api_blueprint.get("/structure")
def serve_structure():
	"""`GET` request endpoint to retrieve structure information for a given protein ID.

	Expects a query parameter `protein_id` in the request URL.

	Loads the `structure_info` DataFrame from the Flask app extensions and checks if the specified protein ID exists.
	Parses the corresponding structure file from `app/structures/PATH`, where `PATH` is specified in the `structure_path`
	column of the structure information DataFrame.
	
	Returns a JSON response containing the contents of the structure file. If `structure_plddt_mean` and/or `structure_ptm`
	columns are present in the structure information DataFrame, the respective values will also be included in the response.

	If the application was started without a `structure_info_path` specified in the configuration, this endpoint will return
	a 501 error indicating that structure information is not available.
	"""
	protein_id = request.args.get("protein_id", None)
	if protein_id is None:
		logger.exception("Missing required request parameter: protein_id")
		return {"error": "Missing required request parameter: protein_id"}, 400

	# Access the structure information DataFrame from the app extensions.
	structure_info_df = _get_extension("structure_info")
	if structure_info_df is None:
		logger.exception("Structure information DataFrame is not available in app extensions.")
		return {"error": "Structure information is not available."}, 501

	# Try to retrieve the structure information for the specified protein ID.
	try:
		structure_info = structure_info_df.loc[protein_id].to_dict()
		with open(structure_info["structure_path"], "r") as f:
			structure_data = f.read()
		structure_info["structure_data"] = structure_data
		del structure_info["structure_path"]
		return structure_info, 200
	except KeyError:
		return {"error": f"No structure was found for {protein_id}."}, 404
	except OSError as exc:
		return {"error": f"Error reading structure file for {protein_id}."}, 500
	except Exception as exc:
		return {"error": f"Unexpected error while retrieving structure information for {protein_id}: {exc}"}, 500


@api_blueprint.get("/predict")
def run_prediction():
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
		return {"error": f"Prediction request failed: {str(e)}"}, 500


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
	nn_pca_distances, nn_indices = kdtree.query(embed_pcs, k=knn)
	nn_pca_distances = nn_pca_distances.tolist() # (m, knn) 2D list.
	nn_indices = nn_indices.tolist() # (m, knn) 2D list.
	
	# Compute weighted t-SNE coordinates based on the distances to the nearest neighbors.
	embed_tsne_coords = [ ] # (m, 2) 2D list.
	nn_tsne_distances = [ ] # Stores the distances of the nearest neighbors in t-SNE space.
	for indices_list, pca_distances_list in zip(nn_indices, nn_pca_distances):
		# Extract the t-SNE coordinates of the nearest neighbors.
		neighbor_tsne_coords = [ tsne["coords"][i] for i in indices_list ]

		# Compute weighted coordinates based on the distances to the nearest neighbors.
		weighted_coords = _weighted_coordinates(neighbor_tsne_coords, pca_distances_list)

		# Recompute the distance in t-SNE space between the weighted coordinates and the nearest neighbor's t-SNE coordinates.
		nn_tsne_distance = round(math.dist(weighted_coords, neighbor_tsne_coords[0]), 3)

		# Append the computed values to the respective lists.
		nn_tsne_distances.append(nn_tsne_distance)
		embed_tsne_coords.append(weighted_coords)

	# Transpose to shape (2, m) for DataFrame construction.
	embed_tsne_coords = np.array(embed_tsne_coords).T
	
	projection_df = pd.DataFrame({
		"protein_ID": embed_protein_ids,
		"tsne_1": embed_tsne_coords[0],
		"tsne_2": embed_tsne_coords[1],
		"nearest_neighbor_ID": [ pca["ids"][_[0]] for _ in nn_indices ],
		"nearest_neighbor_distance": nn_tsne_distances,
	})

	# Merge predictions with projection data and return as JSON.
	result = pd.merge(predictions_df, projection_df, on="protein_ID", how="left")
	return result


def _weighted_coordinates(coordinates: list[list[float]], distances: list[float]) -> list[float]:
	"""Compute a weighted average of coordinates based on distances.
	
	For a data point _P_, the idea is that _P_ was projected into a high-dimensional manifold, e.g. PCA, and
	have its _k_ nearest neighbors in that space. The coordinates of those neighbors in a lower-dimensional space,
	e.g. t-SNE, are known. The goal is to compute a weighted average of those coordinates, where the weights are
	based on the distances to the neighbors in the high-dimensional space.

	This method implements a softmax weighting scheme, where closer neighbors have more influence on the weighted average.
	Weights are computed using a softmax function on the negative distances, so that closer neighbors have more influence
	on the weighted average. The temperature parameter `tau` controls the sharpness of the softmax distribution; smaller
	values of `tau` make the weighting more sensitive to distance differences. Currently a fixed value of `tau = 0.5` is used.

	_Note: If the nearest neighbor (first in the list) has a distance of zero, the function will return the coordinates of
	that neighbor directly, as it is assumed to be the same point in the lower-dimensional space._

	Parameters
	__________
	coordinates (list[list[float]]):
		A list of coordinates of the nearest neighbors in the lower-dimensional space (e.g., t-SNE).
		Each element is a list representing the coordinates of a neighbor.

	distances (list[float]):
		A list of distances to the nearest neighbors in the high-dimensional space (e.g., PCA).
		Each element corresponds to the distance of a neighbor.

	Returns
	_______
	list[float]:
		A list representing the weighted average coordinates in the lower-dimensional space.	
	"""
	# If no coordinates are provided, raise an error.
	if len(coordinates) == 0:
		raise ValueError("No coordinates provided for weighted averaging.")
	# If coordinates and distances lengths do not match, raise an error.
	if len(coordinates) != len(distances):
		raise ValueError("The number of coordinates must match the number of distances.")

	# If the nearest neighbor (first in list) distance is zero, return the corresponding coordinates directly.
	if distances[0] == 0:
		return coordinates[0]

	# Convert distances to weights using softmax.
	distances = np.array(distances)
	tau = .5  # Temperature parameter for softmax; can be adjusted based on desired sensitivity.
	weights = softmax(-distances / tau)  # Invert distances for softmax.

	# Compute weighted average of nearest neighbor coordinates.
	return [
		round( sum(coords[i] * weights[i] for i in range(len(coordinates))), 3 )
		for coords in zip(*coordinates)
	]
