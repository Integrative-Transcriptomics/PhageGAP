"""
API endpoint definitions.
"""

from __future__ import annotations

import logging
import hmac
import math
import pandas as pd
import numpy as np
from scipy.special import softmax
from flask import Blueprint, current_app, request, jsonify
from phagegapclassifier.data import parse_sequence_data, prepare_for_pca
from phagegapclassifier.embed import preprocess_df, compute_embeddings
from phagegapclassifier.pool import pool_embeddings
from phagegapclassifier.predict import run_prediction


logger = logging.getLogger(__name__)
api_blueprint = Blueprint("api", __name__)


def load_app_extension() -> dict:
	"""Return the process-local PhageGap extensions, i.e., models and configurations."""
	try:
		return current_app.extensions["phagegap"]
	except KeyError as exc:
		raise RuntimeError(
			"PhageGap extension was not initialized."
		) from exc


@api_blueprint.before_request
def require_api_token():
	authorization = request.headers.get("Authorization", "")
	scheme, separator, supplied_token = authorization.partition(" ")

	valid = (
		separator
		and scheme.lower() == "bearer"
		and supplied_token
		and hmac.compare_digest(supplied_token, current_app.extensions["phagegap"]["api_token"])
	)

	if not valid:
		response = jsonify({"error": "Unauthorized"})
		response.status_code = 401
		response.headers["WWW-Authenticate"] = "Bearer"
		return response

	return None


@api_blueprint.get("/health")
def health_endpoint():
	return "PhageGap classifier is active.", 200


@api_blueprint.get("/structure")
def structure_endpoint():
	structure_info_df = current_app.extensions["phagegap"]["structure_info"]
	protein_id = request.args.get("protein_id", None)
	if protein_id:
		try:
			structure_info = structure_info_df.loc[protein_id].to_dict()
			with open(structure_info["structure_path"], "r") as f:
				structure_data = f.read()
			structure_info["structure_data"] = structure_data
			del structure_info["structure_path"]
			return jsonify(structure_info), 200
		except KeyError:
			return jsonify({"error": f"Protein ID '{protein_id}' not found in structure database."}), 404
		except OSError as exc:
			return jsonify({"error": f"Error reading structure file for protein ID '{protein_id}'."}), 500
		except Exception as exc:
			return jsonify({"error": f"Unexpected error while retrieving structure for protein ID '{protein_id}': {exc}"}), 500
	else:
		return jsonify({"error": "Missing required query parameter: protein_id"}), 400


@api_blueprint.post("/predict")
def predict_endpoint():
	try:
		# Parse sequence data.
		sequence_text = request.get_data(as_text=True)
		# Run prediction and return results as JSON.
		result = predict(sequence_text)
		return result.to_dict(orient="records"), 200
	except Exception as e:
		logger.exception(f"Prediction request failed: {str(e)}")
		return {"error": f"Prediction request failed: {str(e)}"}, 500


def predict(sequence_text: str) -> pd.DataFrame:
	# Load application extension containing models and configurations.
	extension = load_app_extension()
	config = extension["config"]
	plm = extension["plm"]
	plm_model_type = config.get("embed", {}).get("model_type", None)
	pool_layers = config.get("embed", {}).get("pool_layers", [])
	pool_strategy = config.get("embed", {}).get("pool_strategy", "mean")
	tokenizer = extension["tokenizer"]
	classifier = extension["classifier"]
	label_map = extension["label_map"]
	pca = extension["pca"]
	tsne = extension["tsne"]
	kdtree = extension["kdtree"]

	# Preprocess sequence data.
	df = parse_sequence_data(sequence_text)
	if df.empty:
		raise RuntimeError("Request contains no valid FASTA records.")
	preprocessed_df = preprocess_df(df, plm_model_type)

	# Compute PLM embeddings.
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
	nn_pca_distances, nn_indices = kdtree.query(embed_pcs, k=5)
	nn_pca_distances = nn_pca_distances.tolist() # (m, 5) 2D list.
	nn_indices = nn_indices.tolist() # (m, 5) 2D list.
	
	# Compute weighted t-SNE coordinates based on the distances to the nearest neighbors.
	embed_tsne_coords = [ ] # (m, 2) 2D list.
	nn_tsne_distances = [ ] # Stores the distances of the nearest neighbors in t-SNE space.
	for indices_list, pca_distances_list in zip(nn_indices, nn_pca_distances):
		# Extract the t-SNE coordinates of the nearest neighbors.
		neighbor_tsne_coords = [ tsne["coords"][i] for i in indices_list ]

		# Compute weighted coordinates based on the distances to the nearest neighbors.
		weighted_coords = weighted_coordinates(neighbor_tsne_coords, pca_distances_list)

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


def weighted_coordinates(coordinates: list[list[float]], distances: list[float]) -> list[float]:
	"""Compute a weighted average of coordinates based on distances."""

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
