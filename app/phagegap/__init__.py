from __future__ import annotations

import logging
import requests
import secrets
import pandas as pd
from pathlib import Path
from flask import Flask


# Initialize logging configuration.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app.
logger.debug("Initializing Flask app.")
app = Flask(__name__)

# Only for debugging purposes: Set emulate = True to emulate the behavior of the PhageGap classifier.
# Note: This is by purpose hardcoded to avoid accidental usage in production.
app.extensions["emulate"] = True

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
# Generate a random secret key for session management.
app.config["SECRET_KEY"] = secrets.token_urlsafe(32)

# Load metadata from static resources into a dictionary and validate its contents.
logger.debug("Loading metadata from static resources.")
metadata_path = Path(app.static_folder).joinpath("resources/data/metadata.tsv.gz")
meta_df = pd.read_csv(metadata_path, sep="\t", compression="gzip")
METADATA_COLUMNS = ["protein_ID", "locus_tag", "organism", "phage_ID", "product", "subcategory", "category", "tsne_1", "tsne_2"]
if not all(col in meta_df.columns for col in METADATA_COLUMNS):
	raise RuntimeError(f"Metadata file {str(metadata_path)} is missing required columns. Expected columns: {METADATA_COLUMNS}. Found columns: {list(meta_df.columns)}.")
app.extensions["metadata"] = meta_df

# Load API routes.
logger.debug("Loading API routes.")
from phagegap import api

# Test classifier accessibility if not emulating.
# TODO: We might want to implement a more robust health check for the classifier service in the future.
CLASSIFIER_URLS = [
	"http://134.2.9.250:20101",
	"http://134.2.9.251:20101"
]
if not app.extensions["emulate"]:
	logger.debug("Testing accessibility of the PhageGap classifier service.")
	for url in CLASSIFIER_URLS:
		try:
			response = requests.get(f"{url}/active", timeout=5)
			if response.status_code == 200:
				logger.debug(f"Successfully connected to PhageGap classifier at {url}.")
				app.extensions["classifier_url"] = url
				break
		except requests.RequestException:
			continue
	else:
		logger.error("Failed to access the PhageGap classifier. The service might be down or unreachable.")
		raise RuntimeError("Failed to access the PhageGap classifier. Please check the service status and network connectivity.")
else:
	logger.debug("Emulating PhageGap classifier behavior. No actual requests will be sent to the classifier service.")

# Initialize SocketIO.
# TODO: This has to be adjustes to allow communication with the PhageGap classifier.
#from flask_socketio import SocketIO
#socketio = SocketIO(app, cors_allowed_origins="*", manage_session=False)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5001)
	#socketio.run(app)
