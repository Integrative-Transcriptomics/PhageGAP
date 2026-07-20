from __future__ import annotations

import logging
import tomllib
from typing import Any

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

	# TODO: Why is the checkpoint the PLM model name?
	checkpoint = embed_config.get("model_type")
	logger.info("Loading embedding model %s on %s.", checkpoint, device)

	plm, tokenizer = monitor_load_embed_model(checkpoint, device)

	model_path = predict_config.get("model_path")
	model_type = predict_config.get("model_type")
	# TODO: What exactly is the model number and why is it stored in the file?
	model_number = extract_model_number(model_path)

	logger.info("Loading classifier %s on %s.", model_type, device)

	classifier, label_map, _ = monitor_load_predict_model(
		model_path,
		model_number,
		model_type,
	)

	logger.info("Loading PCA and t-SNE objects.")

	pca = joblib.load(manifold_config.get("pca_path"))
	tsne = joblib.load(manifold_config.get("tsne_path"))

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
	app.extensions["phagegap_models"] = {
		"config": config,
		"device": device,
		"checkpoint": checkpoint,
		"plm": plm,
		"tokenizer": tokenizer,
		"classifier": classifier,
		"label_map": label_map,
		"pca": pca,
		"tsne": tsne,
	}

	# Register API blueprint.
	from phagegapclassifier.api import api_blueprint

	app.register_blueprint(api_blueprint)

	logger.info("PhageGap application initialized on %s.", device)

	return app


app = create_app()
