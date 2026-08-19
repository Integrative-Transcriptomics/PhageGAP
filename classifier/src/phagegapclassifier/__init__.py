from __future__ import annotations

import logging
import tomllib
import joblib
import torch
import pandas as pd
from typing import Any
from pathlib import Path
from flask import Flask
from scipy.spatial import KDTree
from phagegapclassifier.embed import load_embed_model, MODEL_CONFIGS
from phagegapclassifier.predict import load_predict_model
from phagegapclassifier.utils import set_determinism, extract_model_number


# Initialize logging configuration.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

def load_config() -> dict[str, Any]:
	"""Load application configurations from local `config.toml` file.
	
	The configuration file is expected to be located in the same directory as this script.
	It should contain the following sections and keys:

	[app]
	_____
	- max_content_length: The maximum size (in bytes) of incoming request data. Default is 500,000 bytes.
	- max_form_memory_size: The maximum size (in bytes) of form data in memory. Default is 500,000 bytes.
	- max_form_parts: The maximum number of parts in a multipart/form-data request. Default is 1,000.
	- api_token_path: Path to the file containing a secure token for authentication. If not specified, the application will not use token authentication.
	- structure_info_path: Path to a tab-delimited file containing protein structure information. If not specified, the application will not provide structure information.

	[setup]
	_______
	- seed: Random seed for reproducibility. Default is 0.
	- device: The device to use for computations (`cpu` or `cuda`). If `cuda` is specified but unavailable, the application will fall back to CPU.

	[embed]
	_______
	- model_type: The type of embedding model to use. Currently only "prot_t5", "prot_t5xxl", "prost_t5" are supported. Default is "prot_t5".
	- pool_layers: The layers to pool from the embedding model. Default is [] (all).
	- pool_strategy: The pooling strategy to use for embeddings. Options include "mean", "max", "FC". Default is "mean".

	[predict]
	_________
	- model_path: Path to the saved model pickle file for predictions (**mandatory**).
	- model_type: The type of prediction model to use. Only "cnn" and "cnn_mlp" are supported. Default is "cnn".

	[manifold]
	__________
	- pca_path: Path to the saved PCA object for dimensionality reduction (**mandatory**).
	- tsne_path: Path to the saved t-SNE object for dimensionality reduction (**mandatory**).
	- knn: The number of nearest neighbors to consider for t-SNE coordinate projection. Default is 3.
	"""
	with open("config.toml", "rb") as file:
		return tomllib.load(file)


def select_device(device: str) -> torch.device:
	"""Select the appropriate device (CPU or CUDA) based on configuration and availability.
	
	Parameters
	__________
	device (str):
		The device specified in the configuration. Should be either "cpu" or "cuda".

	Returns
	_______
	torch.device:
		The selected device for computations.
	"""
	if device == "cuda":
		if torch.cuda.is_available():
			logger.info("CUDA is available; using GPU for computations.")
			return torch.device("cuda")
		else:
			logger.warning("CUDA was requested but is unavailable; using CPU for computations.")
			# No need to return here; will fall through to return CPU below.
	logger.info("Using CPU for computations.")
	return torch.device("cpu")


def create_app() -> Flask:
	"""Create and configure the PhageGAP classifier Flask application.
	
	Returns
	_______
	Flask:
		The configured Flask application instance.
	"""
	config = load_config()

	setup_config = config.get("setup", {})
	app_config = config.get("app", {})
	embed_config = config.get("embed", {})
	predict_config = config.get("predict", {})
	manifold_config = config.get("manifold", {})

	set_determinism(setup_config.get("seed", 0))

	device = select_device(setup_config.get("device", "cpu"))

	embed_model_type = embed_config.get("model_type", "prot_t5")
	# Check if a supported pLM type is specified in the configuration.
	if embed_model_type not in MODEL_CONFIGS:
		raise RuntimeError(f"Unsupported pLM type: {embed_model_type}. Supported types are: {list(MODEL_CONFIGS.keys())}.")
	logger.info("Loading embedding model %s on %s.", embed_model_type, device)
	plm, tokenizer = load_embed_model(embed_model_type, device)

	predict_model_path = predict_config.get("model_path")
	if predict_model_path is None:
		raise RuntimeError("Classifier model path is not specified in the configuration.")

	predict_model_type = predict_config.get("model_type", "cnn")
	if predict_model_type not in ("cnn", "cnn_mlp"):
		raise ValueError(f"Unsupported classifier model type: {predict_config.get('model_type')}. Supported types are 'cnn' and 'cnn_mlp'.")
	
	predict_model_number = extract_model_number(predict_model_path)
	logger.info("Loading classifier %s on %s.", predict_model_type, device)
	classifier, label_map, _ = load_predict_model(
		predict_model_path,
		predict_model_number,
		predict_model_type,
	)

	logger.info("Loading PCA and t-SNE objects.")
	pca_path = manifold_config.get("pca_path")
	if pca_path is None:
		raise RuntimeError("PCA path is not specified in the configuration.")
	pca = joblib.load(manifold_config.get("pca_path"))
	tsne_path = manifold_config.get("tsne_path")
	if tsne_path is None:
		raise RuntimeError("t-SNE path is not specified in the configuration.")
	tsne = joblib.load(manifold_config.get("tsne_path"))

	knn = manifold_config.get("knn", 3)
	logger.info(f"Building KDTree for nearest neighbor (k={knn}) search.")
	if pca["coords"].shape[0] == 0:
		raise RuntimeError("PCA coordinates are empty; cannot build KDTree.")
	kdtree = KDTree(pca["coords"])

	logger.info("Loading structure information.")
	# See function documentation for expected format of structure_info_path file.
	structure_info_path = app_config.get("structure_info_path")
	if structure_info_path is None:
		logger.warning("Structure information path is not specified; the application will not provide structure information.")
		structure_info_df = None
	else:
		structure_info_df = pd.read_csv(app_config.get("structure_info_path", None), delimiter="\t")
		for col in ["protein_ID", "structure_path"]:
			if col not in structure_info_df.columns:
				raise RuntimeError(f"Missing required column '{col}' in structure information file.")
		structure_info_df.set_index("protein_ID", inplace=True)

	logger.info("Loading API token.")
	api_token_path = app_config.get("api_token_path")
	if api_token_path is None:
		logger.warning("API token path is not specified; the application will not use token authentication.")
		api_token = None
	else:
		api_token = Path(api_token_path).read_text(encoding="utf-8").strip()
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
		"kdtree": kdtree,
		"knn": knn,
		"tsne": tsne,
		"api_token": api_token,
		"structure_info": structure_info_df,
	}

	# Register API blueprint.
	from phagegapclassifier.api import api_blueprint

	app.register_blueprint(api_blueprint)

	logger.info("PhageGap application initialized on %s.", device)

	return app


# Initialize the Flask application when this module is imported.
app = create_app()
