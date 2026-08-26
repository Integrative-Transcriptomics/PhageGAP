from __future__ import annotations

import logging
import secrets
import os
import pandas as pd
from pathlib import Path
from flask import Flask


# Initialize logging configuration.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app.
logger.debug("Initializing Flask app.")
app = Flask(__name__)

# Only for debugging purposes: Set emulate = True to emulate the behavior of the PhageGAP classifier.
try :
	app.extensions["emulate"] = int(os.getenv("EMULATE", 0)) == 1
except ValueError:
	logger.warning("Invalid value for EMULATE environment variable. Defaulting to False.")
	app.extensions["emulate"] = False

# Load API token.
logger.debug("Loading API token from static resources.")
api_token = None
try:
	# TODO: We might want to change how we load the API token in the future, e.g., by using environment variables or a configuration file.
	api_token_path = Path(app.static_folder).joinpath("resources/secrets/token")
	api_token = api_token_path.read_text(encoding="utf-8").strip()
	if len(api_token) < 32:
		logger.warning("API token should contain at least 32 characters for security reasons.")
except OSError as exc:
	logger.warning( f"Failed to read API token from {str(api_token_path)}. "
					"API endpoints will be accessible outside of the application context and"
					"communication with the classifier might fail." )
app.extensions["api_token"] = api_token

# Configure Flask app.
app.json.sort_keys = False # Preserve order of JSON keys in responses from server.
app.config["MAX_CONTENT_LENGTH"] = 500_000
app.config["MAX_FORM_MEMORY_SIZE"] = 500_000
app.config["MAX_FORM_PARTS"] = 1000
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_NAME"] = "phagegap_session"

# Load secret key for session signing.
api_key = None
try:
	api_key_path = Path(app.static_folder).joinpath("resources/secrets/key")
	api_key = api_key_path.read_text(encoding="utf-8").strip()
	if len(api_key) < 32:
		logger.warning("Secret key should contain at least 32 characters for security reasons.")
except OSError as exc:
	logger.warning( f"Failed to read secret key from {str(api_key_path)}. "
					"Session management will be insecure and might fail." )
app.config["SECRET_KEY"] = api_key if api_key is not None else secrets.token_hex(32)

# Load metadata from static resources into a dictionary and validate its contents.
logger.debug("Loading metadata from static resources.")
metadata_path = Path(app.static_folder).joinpath("resources/data/metadata.tsv.gz")
metadata_df = pd.read_csv(metadata_path, sep="\t", compression="gzip")
METADATA_COLUMNS = ["protein_ID", "locus_tag", "organism", "phage_ID", "product", "subcategory", "category", "tsne_1", "tsne_2"]
if not all(col in metadata_df.columns for col in METADATA_COLUMNS):
	raise RuntimeError(f"Metadata file {str(metadata_path)} is missing required columns. Expected columns: {METADATA_COLUMNS}. Found columns: {list(metadata_df.columns)}.")
app.extensions["metadata"] = metadata_df

# Load structure information from static resources into a DataFrame and validate its contents.
logger.debug("Loading structure information from static resources.")
structure_info_path = Path(app.static_folder).joinpath("resources/data/structures.tsv.gz")
if structure_info_path is None:
	logger.warning("Structure information path is not specified; the application will not provide structure information.")
	structure_info_df = None
else:
	structure_info_df = pd.read_csv(structure_info_path, delimiter="\t", compression="gzip")
	for col in ["protein_ID", "structure_path", "structure_plddt_mean", "structure_ptm"]:
		if col not in structure_info_df.columns:
			raise RuntimeError(f"Missing required column '{col}' in structure information file.")
	structure_info_df.set_index("protein_ID", inplace=True)
app.extensions["structure_info"] = structure_info_df

# Load API routes.
logger.debug("Loading API routes.")
from phagegap import api

# Initialize SocketIO.
# TODO: This has to be adjustes to allow communication with the PhageGAP classifier.
#from flask_socketio import SocketIO
#socketio = SocketIO(app, cors_allowed_origins="*", manage_session=False)
