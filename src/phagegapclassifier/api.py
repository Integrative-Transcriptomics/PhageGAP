"""
API endpoint definitions.
"""

from __future__ import annotations

import logging
import hmac

from flask import Blueprint, current_app, request, jsonify

from phagegapclassifier.data import parse_sequence_data, prepare_for_pca
from phagegapclassifier.embed import preprocess_df, compute_embeddings
from phagegapclassifier.pool import pool_embeddings
from phagegapclassifier.predict import predict as run_prediction


logger = logging.getLogger(__name__)
api_blueprint = Blueprint("api", __name__)


def get_resources() -> dict:
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
	return "PhageGap Classifier is active.", 200


@api_blueprint.post("/predict")
def predict_endpoint():
	return "Not implemented yet.", 501
	try:
		resources = get_resources()

		config = resources["config"]
		checkpoint = resources["checkpoint"]
		plm = resources["plm"]
		tokenizer = resources["tokenizer"]
		classifier = resources["classifier"]
		label_map = resources["label_map"]
		pca = resources["pca"]
		tsne = resources["tsne"]

		# Parse and preprocess sequence data.
		df = parse_sequence_data(request.get_data())
		preprocessed_df = preprocess_df(df, checkpoint)

		# Compute PLM embeddings.
		embed_dict = compute_embeddings(
			checkpoint,
			preprocessed_df,
			plm,
			tokenizer,
			True,
		)

		embed_config = config.get("embed", {})

		embed_dict_pooled = pool_embeddings(
			embed_dict,
			checkpoint,
			layers=embed_config.get("pool_layers", []),
			strategy=embed_config.get("pool_strategy", "mean"),
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

		# Retrieve precomputed manifold data.
		pca_model = pca["model"]
		pca_coords = pca["coords"]
		pca_ids = pca["ids"]

		tsne_embedding = tsne["coords"]
		tsne_ids = tsne["ids"]

		# Project the new samples.
		X_new, ids_new = prepare_for_pca(embed_dict_pooled)
		coords_new = pca_model.transform(X_new)
		tsne_coords_new = tsne_embedding.transform(coords_new)

		# Replace or extend this response as needed.
		return {
			"predictions": predictions_df.to_dict(orient="records"),
			"ids": list(ids_new),
			"pca_coordinates": coords_new.tolist(),
			"tsne_coordinates": tsne_coords_new.tolist(),
		}, 200

	except Exception:
		logger.exception("Prediction request failed.")
		return {"error": "Prediction request failed."}, 500
