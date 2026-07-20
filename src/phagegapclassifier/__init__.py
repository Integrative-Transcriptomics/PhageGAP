import logging, joblib, torch, tomllib
from __future__ import annotations
from flask import Flask
from flask.sessions import NullSessionInterface
from dotenv import load_dotenv
from utils import set_determinism
from data import extract_model_number
from embed import monitor_load_embed_model
from predict import monitor_load_predict_model


# Initialize logging configuration.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Load system variables and declare helpers.
load_dotenv()

# Load project configuration from config.toml.
with open("config.toml", "r") as f:
    config = tomllib.load(f)

# Set determinism and environment.
set_determinism(config.get('setup').get('seed', 0))
if config.get('setup').get('device', 'cpu') == "cuda" and torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")
	if config.get('setup').get('device', 'cpu') == "cuda":
		logging.info("CUDA not available, using CPU.")

# Get model number.
# TODO: What exactly is the model number and why is it stored in the file?
model_no = extract_model_number(config.get('predict').get('model_path', None))

# Instantiate PLM.
# TODO: Why is the checkpoint the PLM model name?
checkpoint = config.get('embed').get('model_type', None)
logger.info(f"Instantiating model {checkpoint} on {device}.")
plm, tokenizer = monitor_load_embed_model(checkpoint, device)

# Instantiate CNN.
logger.info(f"Instantiating model {config.get('predict').get('model_type', None)} on {device}.")
classifier, label_map, _ = monitor_load_predict_model(
	config.get('predict').get('model_path', None),
	model_no,
	config.get('predict').get('model_type', None)
)

# Instantiate PCA and t-SNE manifold objects.
logger.info(f"Instantiating PCA and t-SNE objects.")
pca = joblib.load(config.get('manifold').get('pca_path', None))
tsne = joblib.load(config.get('manifold').get('tsne_path', None))

class NoSessionFlask(Flask):
    session_interface = NullSessionInterface()

# Initialize the Flask application without a session interface.
logger.info(f"Initialize Flask application.")
app = NoSessionFlask(__name__)

# Set Flask application parameters.
app.config["MAX_CONTENT_LENGTH"] = config.get('app').get('max_content_length', 500_000)
app.config["MAX_FORM_MEMORY_SIZE"] = config.get('app').get('max_form_memory_size', 500_000)
app.config["MAX_FORM_PARTS"] = config.get('app').get('max_form_memory_size', 1000)

from phagegapclassifier import api

if __name__ == "__main__":
	# Start the Flask application.
	app.run(
		debug=config.get('app').get('debug', False),
		host=config.get('app').get('host', "0.0.0.0"),
		port=config.get('app').gett('port', 5001)
	)
