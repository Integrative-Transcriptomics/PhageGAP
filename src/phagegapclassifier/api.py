"""
API endpoint definitions.
"""

from __future__ import annotations

import logging
import hmac

import pandas as pd

from flask import Blueprint, current_app, request, jsonify
from phagegapclassifier.data import parse_sequence_data, prepare_for_pca
from phagegapclassifier.embed import preprocess_df, compute_embeddings
from phagegapclassifier.pool import pool_embeddings
from phagegapclassifier.predict import run_prediction


logger = logging.getLogger(__name__)
api_blueprint = Blueprint("api", __name__)


def get_extension() -> dict:
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


@api_blueprint.post("/predict")
def predict_endpoint():
	try:
		extension = get_extension()

		config = extension["config"]
		plm = extension["plm"]
		plm_model_type = config.get("embed", {}).get("model_type", None)
		pool_layers = config.get("embed", {}).get("pool_layers", [])
		pool_strategy = config.get("embed", {}).get("pool_strategy", "mean")
		tokenizer = extension["tokenizer"]
		classifier = extension["classifier"]
		label_map = extension["label_map"]
		pca = extension["pca"]
		pca_kdtree = extension["pca_kdtree"]
		tsne = extension["tsne"]

		# Parse and preprocess sequence data.
		sequence_text = request.get_data(as_text=True)
		df = parse_sequence_data(sequence_text)
		if df.empty:
			return {
				"error": "Request contains no valid FASTA records."
			}, 400
		preprocessed_df = preprocess_df(df, plm_model_type)

		# Compute PLM embeddings.
		embed_dict = compute_embeddings(
			plm_model_type,
			preprocessed_df,
			plm,
			tokenizer,
			True,
		)

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

		predictions_df.drop(
			columns=["processed_seq"],
			inplace=True,
			errors="ignore",
		)

		# Project the new samples and construct a DataFrame.
		embed_matrix, embed_protein_ids = prepare_for_pca(embed_dict_pooled)
		embed_pcs = pca["model"].transform(embed_matrix)

		# Search for nearest neighbor (nn) (k=1) in the PCA space.
		nn_pc_distances, nn_indices = pca_kdtree.query(embed_pcs) # k=1 and p=2 are defaults.
		nn_pc_distances = nn_pc_distances.tolist()
		nn_indices = nn_indices.tolist()

		embed_tsne_coords = tsne["coords"].transform(embed_pcs).tolist()
		projection_df = pd.DataFrame({
			"protein_ID": embed_protein_ids,
			"tsne_coordinates": embed_tsne_coords,
			"nearest_neighbor_ID": [ pca["ids"][i] for i in nn_indices ],
			"nearest_neighbor_distance": nn_pc_distances,
		})

		# Merge predictions with projection data and return as JSON.
		result = pd.merge(predictions_df, projection_df, on="protein_ID", how="left")
		return result.to_dict(orient="records"), 200

	except Exception:
		logger.exception("Prediction request failed.")
		return {"error": "Prediction request failed."}, 500
