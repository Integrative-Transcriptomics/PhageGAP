from __future__ import annotations

import logging
import tomllib
from typing import Any
from pathlib import Path

import joblib
import torch
from dotenv import load_dotenv
from flask import Flask

from phagegapclassifier.data import extract_model_number
from phagegapclassifier.embed import monitor_load_embed_model
from phagegapclassifier.predict import monitor_load_predict_model
from phagegapclassifier.utils import set_determinism


# Initialize logging configuration.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict[str, Any]:
	"""Load configuration from config.toml and .env."""
	load_dotenv()

	with open("config.toml", "rb") as file:
		return tomllib.load(file)


def select_device(config: dict[str, Any]) -> torch.device:
	"""Select the appropriate device (CPU or CUDA) based on configuration and availability."""
	setup_config = config.get("setup", {})
	requested_device = setup_config.get("device", "cpu")
	
	logger.info("CUDA is available: %s", torch.cuda.is_available())

	if requested_device == "cuda":
		if torch.cuda.is_available():
			return torch.device("cuda")

		logger.warning("CUDA was requested but is unavailable; using CPU.")

	return torch.device("cpu")


def create_app() -> Flask:
	"""Create and configure the Flask application."""
	config = load_config()

	setup_config = config.get("setup", {})
	app_config = config.get("app", {})
	embed_config = config.get("embed", {})
	predict_config = config.get("predict", {})
	manifold_config = config.get("manifold", {})

	set_determinism(setup_config.get("seed", 0))
	device = select_device(config)

	embed_model_type = embed_config.get("model_type")
	logger.info("Loading embedding model %s on %s.", embed_model_type, device)
	plm, tokenizer = monitor_load_embed_model(embed_model_type, device)

	predict_model_path = predict_config.get("model_path")
	predict_model_type = predict_config.get("model_type")
	predict_model_number = extract_model_number(predict_model_path)

	logger.info("Loading classifier %s on %s.", predict_model_type, device)
	classifier, label_map, _ = monitor_load_predict_model(
		predict_model_path,
		predict_model_number,
		predict_model_type,
	)

	logger.info("Loading PCA and t-SNE objects.")
	pca = joblib.load(manifold_config.get("pca_path"))
	tsne = joblib.load(manifold_config.get("tsne_path"))

	logger.info("Loading API token.")
	try:
		token_path = Path(app_config.get("api_token_path", None))
	except TypeError as exc:
		raise RuntimeError(
			"API token path is not specified in the configuration."
		) from exc
	try:
		api_token = token_path.read_text(encoding="utf-8").strip()
	except OSError as exc:
		raise RuntimeError(
			f"Could not read API token from {token_path}."
		) from exc
	if len(api_token) < 32:
		raise RuntimeError("API token must contain at least 32 characters.")

	app = Flask(__name__)

	app.config.update(
		MAX_CONTENT_LENGTH=app_config.get(
			"max_content_length",
			500_000,
		),
		MAX_FORM_MEMORY_SIZE=app_config.get(
			"max_form_memory_size",
			500_000,
		),
		MAX_FORM_PARTS=app_config.get(
			"max_form_parts",
			1_000,
		),
	)

	# Store process-local runtime resources in the Flask application.
	app.extensions["phagegap"] = {
		"config": config,
		"device": device,
		"plm": plm,
		"tokenizer": tokenizer,
		"classifier": classifier,
		"label_map": label_map,
		"pca": pca,
		"tsne": tsne,
		"api_token": api_token,
	}

	# Register API blueprint.
	from phagegapclassifier.api import api_blueprint

	app.register_blueprint(api_blueprint)

	logger.info("PhageGap application initialized on %s.", device)

	return app


app = create_app()
