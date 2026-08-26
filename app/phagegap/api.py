"""
API endpoints for the `phagegap` Flask app.
"""

from __future__ import annotations

import hmac
import secrets
import requests
import logging
import requests
import numpy as np
from phagegap import app
from phagegap.util import GFFParser, read_sequence_from_cif, align_sequences
from flask import request, render_template, jsonify, session, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from traceback import print_exc
from pathlib import Path
from Bio import SeqIO
from io import StringIO
from scipy.special import softmax


logger = logging.getLogger(__name__)

# Initialize Flask-Limiter for rate limiting API requests.
limiter = Limiter(key_func=get_remote_address, app=app)

# Define the URL of the PhageGAP classifier service.
# TODO: This needs to be queried before each request in the future.
CLASSIFIER_URL = "http://134.2.9.251:20101"


@app.before_request
def authenticate():
	"""Checks for a valid cookie in the request headers for API endpoints.

	If the cookie is invalid, return a 401 Unauthorized response.
	"""
	if not request.path.startswith("/api/"):
		return None
	
	# 1. Users need to accept the cookie policy to use the API.
	consent_cookie = request.cookies.get("phagegap_consent")

	# 2. Users need to provide a valid CSRF token in the request headers.
	expected_csrf_token = session.get("csrf_token", "0")
	supplied_csrf_token = request.headers.get("X-CSRF-Token", "1")

	valid = ( consent_cookie == "accepted" and hmac.compare_digest(supplied_csrf_token, expected_csrf_token) )

	if not valid:
		return Response(
			"Unauthorized request. Missing or invalid token.",
			401,
			{
				"WWW-Authenticate": "Bearer",
			}
		)

	return None


@app.route("/")
def index():
	"""Renders the main index page of the PhageGap web application.
	"""
	if "csrf_token" not in session:
		session["csrf_token"] = secrets.token_urlsafe(32)
	return render_template("index.html", csrf_token=session["csrf_token"])


@app.route("/api/metadata", methods=["GET"])
def serve_metadata():
	"""Serves the metadata as a JSON response.
	
	This endpoint is used to provide metadata information to the client-side application.
	"""
	try:
		return app.extensions["metadata"].fillna("null").to_dict(orient="records"), 200
	except Exception as e:
		print_exc()
		return f"Failed to fetch metadata: {str(e)}", 500


@app.route("/api/classifier/nninfo", methods=["POST"])
def serve_nearest_neighbor_information():
	"""Serves information about the nearest neighbor of a user-provided protein.

	Expects the following parameters:
	- `nn_protein_id`: The protein ID of the nearest neighbor as argument in the query string.
	- `protein_seq`: The amino acid sequence of the user-provided protein as form data.

	Returns a JSON response containing the structure data and sequence alignment information for
	the nearest neighbor protein, along with optional pLDDT and predicted TM-score if available.

	Response
	________
	- JSON object containing the structure data and sequence alignment information:
	
	structure_data:
		The structure data of the nearest neighbor protein in CIF format.
	
	sequence_alignment:
		A list containing the aligned sequences of the nearest neighbor and the user-provided protein.
	
	plddt_mean (optional):
		The mean pLDDT score of the nearest neighbor structure, if available.
	
	ptm (optional):
		The predicted TM-score of the nearest neighbor structure, if available.
	"""
	try:
		if not "nn_protein_id" in request.args:
			return "No nearest neighbor protein ID was provided.", 400
		if not "protein_seq" in request.form:
			return "No protein sequence was provided.", 400
		nn_protein_id = request.args.get("nn_protein_id")
		protein_seq = request.form.get("protein_seq")

		structure_info = _get_protein_structure(nn_protein_id)
		if isinstance(structure_info, tuple):
			# If structure_info is a tuple, it contains an error message and status code.
			return structure_info[0], structure_info[1]

		# Extract the amino acid sequence from the CIF content.
		cif_content = structure_info.get("structure_data")
		if cif_content is None:
			return f"No structure data found for {nn_protein_id}.", 404
		nn_seq = read_sequence_from_cif(cif_content)

		# Align the nearest neighbor sequence with the sequence provided by the user.
		structure_info["sequence_alignment"] = list(align_sequences(nn_seq, protein_seq))

		return structure_info, 200
	except Exception as e:
		print_exc()
		return f"Failed to retrieve nearest neighbor information: {str(e)}", 500


@app.route("/api/gff", methods=["POST"])
@limiter.limit("10 per minute")  # Limit to 50 requests per minute per IP address.
def process_gff():
	"""Processes a GFF file uploaded by the user and returns the parsed features as JSON.

	See :func:`GFFParser.parse` for details on the returned data structure.
	"""
	try:
		if "file" not in request.files:
			return {"error": "No GFF file provided."}, 400
		gff_text = request.files["file"].read().decode("utf-8")
		gff_parser = GFFParser(StringIO(gff_text))
		features = gff_parser.parse()
		# Return parsed features as JSON.
		return jsonify(features), 200
	except Exception as e:
		print_exc()
		return f"Failed to process GFF: {str(e)}", 500


@app.route("/api/classifier/predict", methods=["POST"])
@limiter.limit("10 per minute")  # Limit to 50 requests per minute per IP address.
def serve_classifier_prediction():
	"""Serves the prediction results from the PhageGap classifier for a given set of protein sequences.

	Expects at least one of following parameters:
	- `file`: A FASTA file containing the protein sequences to be classified (optional).
	- `text`: A text input containing protein sequences in FASTA format (optional).

	Returns a JSON response containing the classification results for each protein sequence.
	"""
	try:
		if "file" not in request.files and "text" not in request.form:
			return "No sequence content in request.", 400

		# Collect FASTA input.
		parts = []
		if "file" in request.files:
			file_text = request.files["file"].read().decode("utf-8")
			parts.append(file_text.strip())
		if "text" in request.form:
			text = request.form["text"]
			parts.append(text.strip())
		sequence_text = "\n".join(parts) + "\n"

		# Parse sequences once for validation.
		sequence_records = []
		try:
			sequence_records = list(
				SeqIO.parse(StringIO(sequence_text), "fasta")
			)
		except Exception as e:
			return f"Error parsing FASTA sequences: {e}", 500

		# Check if any valid FASTA records were found.
		if len(sequence_records) == 0:
			return "No valid FASTA records found.", 400

		# Prepare user data structure to store results.
		# Note: It is important to duplicate the protein ID as key for later merging.
		user_results = {
			"records": {
				record.id: {
					"protein_ID": record.id,
					"description": record.description,
					"sequence": str(record.seq)
				}
				for record in sequence_records
			},
		}

		''' TODO: Legacy code that uses SocketIO for communication with client.
		socketio.emit(
			"notify",
			{
				"title": "Classification Started",
				"message": (
					f"Started classification of "
					f"{len(sequence_records)} sequences."
				),
			},
		)
		'''

		if app.extensions["emulate"]:
			response_data = _emulate_classifier_response(user_results)
		else :
			try:
				# Forward the original FASTA text to the classifier apptainer.
				response = requests.get(
					f"{CLASSIFIER_URL}/predict",
					data=sequence_text.encode("utf-8"),
					headers={
						"Authorization": (
							f"Bearer {app.extensions['api_token']}"
						),
						"Content-Type": "text/x-fasta; charset=utf-8",
						"Accept": "application/json",
					},
					timeout=(2,1200),
				)
				response.raise_for_status()
				response_data = response.json()
			except requests.ConnectionError:
				return "The PhageGAP classifier service is not accessible. Please try again later.", 404
			except requests.Timeout:
				return "The PhageGAP classifier service did not respond in time. The service may be busy, or the time limit for processing your request may have been exceeded.", 504
			except requests.HTTPError as exc:
				return f"HTTP error {exc.response.status_code} from the PhageGAP classifier service: {exc.response.text}", exc.response.status_code

			# Merge the results from the classifier with the user data.
			for record in response_data["predictions"]:
				protein_id = record.get("protein_ID")
				if protein_id in user_results["records"]:
					# TODO: Keys need to be adjusted, if changed in classifier.
					for key in ["top1", "P(top1)", "top2", "P(top2)", "top3", "P(top3)"]:
						user_results["records"][protein_id][key] = record.get(key)

		# Convert the user data records to a list for JSON serialization.
		user_results = list(user_results["records"].values())
		nearest_neighbors = response_data.get("nearest_neighbors", {})

		# Process the nearest neighbor information to estimate weighted coordinates and distances.
		_estimate_tsne_coords(user_results, nearest_neighbors)

		# Return the results to the client.
		return {
			"predictions": user_results,
			"nearest_neighbors": nearest_neighbors,
		}, 200

	except Exception as e:
		print_exc()
		return f"An unexpected error occurred while processing the classification request: {str(e)}", 500


def _get_protein_structure(protein_id: str) -> dict|tuple:
	"""Retrieves the structure information for a given protein ID from the app extensions.
	
	This requires that the structure information DataFrame is available in the app extensions,
	as well as the corresponding structure files mounted in `/app/structures/`.

	If the structure information is not available, returns an error message and status code.

	Parameters
	__________
	protein_id (str):
		The protein ID for which to retrieve structure information.

	Returns
	_______
	dict:
		A dictionary containing the structure information and data for the specified protein ID.
	
	or
	
	tuple:
		A tuple containing an error message and an HTTP status code if the structure information is not available
	"""

	# Access the structure information DataFrame from the app extensions.
	structure_info_df = app.extensions["structure_info"]
	if app.extensions["structure_info"] is None:
		return "Structure information is not available in the application.", 404

	# Try to retrieve the structure information for the specified protein ID.
	try:
		structure_info = structure_info_df.loc[protein_id].to_dict()
		if app.extensions["emulate"]:
			# TODO: This is only needed locally for testing, but should be removed in production.
			structure_info_path = Path(app.static_folder).joinpath("resources/data/emulate_nn_structure.cif")
			structure_info["structure_data"] = structure_info_path.read_text(encoding="utf-8")
		else:
			with open(f"/app/{structure_info['structure_path']}", "r") as f:
				structure_data = f.read()
			structure_info["structure_data"] = structure_data

		# Remove the structure path from the response, as it is not needed by the client.
		del structure_info["structure_path"]
		return structure_info
	except KeyError:
		return f"No structure was found for {protein_id}.", 404
	except OSError as exc:
		return f"Error reading structure file for {protein_id}.", 500
	except Exception as exc:
		return f"Unexpected error while retrieving structure information for {protein_id}: {exc}", 500


def _emulate_classifier_response(user_data: dict) -> dict:
	"""Emulates the response of the PhageGAP classifier for testing purposes.

	Parameters
	__________
	user_data (dict):
		A dictionary containing the user-provided protein sequences and their associated data.

	Returns
	_______
	dict:
		A dictionary containing the emulated predictions and nearest neighbor information for each protein sequence.
	"""
	# Access the metadata DataFrame from the app extensions.
	metadata_df = app.extensions["metadata"]

	# Init. nearest neighbor information dictionary.
	nearest_neighbors = {}

	for record in user_data["records"].values():
		record_nearest_neighbors = []

		# Draw a random sample of 5 nearest neighbor protein IDs from the metadata DataFrame.
		sample = metadata_df.sample(n=5)

		# Generate five PCA distances from lognormal distribution with mean=2 and sigma=1.
		pca_distances = np.random.lognormal( mean=2, sigma=1, size=5 )

		# Assign each randomly selected nearest neighbor a PCA distance.
		for i, (_, row) in enumerate(sample.iterrows()):
			nearest_neighbor = {
				"protein_ID": row["protein_ID"],
				"pca_distance": pca_distances[i],
				"tsne_1": row["tsne_1"],
				"tsne_2": row["tsne_2"],
				"subcategory": row["subcategory"],
			}
			record_nearest_neighbors.append(nearest_neighbor)

		# Sort nearest neighbors by PCA distance.
		record_nearest_neighbors.sort(key=lambda x: x["pca_distance"])

		# Emulate classification results by using the 'subcategory' field of the three nearest neighbors as the top three predicted classes.
		probabilities = np.random.exponential(2, size=4)
		probabilities.sort()
		probabilities = probabilities[::-1]  # Sort in descending order.
		probabilities /= np.sum(probabilities)
		record["top1"] = record_nearest_neighbors[0]["subcategory"]
		record["P(top1)"] = probabilities[0]
		record["top2"] = record_nearest_neighbors[1]["subcategory"]
		record["P(top2)"] = probabilities[1]
		record["top3"] = record_nearest_neighbors[2]["subcategory"]
		record["P(top3)"] = probabilities[2]

		# Delete the 'subcategory' field from the nearest neighbor records to avoid redundancy.
		for nn in record_nearest_neighbors:
			del nn["subcategory"]

		# Assign the nearest neighbor information to the main 'nearest_neighbors' dictionary.
		nearest_neighbors[record["protein_ID"]] = record_nearest_neighbors

	# Return the emulated classifier response as a dictionary containing the predictions and nearest neighbor information.
	return {
		"predictions": user_data["records"],
		"nearest_neighbors": nearest_neighbors,
	}


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


def _estimate_tsne_coords(user_data: list, nearest_neighbors: dict):
	"""In-place modification of the provided user data (list of records) to estimate t-SNE coordinates based on the nearest neighbor information.

	Parameters
	__________
	user_data (list):
		A list of dictionaries, each representing a protein record with its associated data.
		<pre>
			"protein_ID": str, the identifier of the protein.
			"description": str, the description of the protein.
			"sequence": str, the amino acid sequence of the protein.
			"top1": str, the (1st) predicted class label for the protein.
			"top2": str, the (2nd) predicted class label for the protein.
			"top3": str, the (3rd) predicted class label for the protein.
			"P(top1)": float, the probability of the (1st) predicted class label.
			"P(top2)": float, the probability of the (2nd) predicted class label.
			"P(top3)": float, the probability of the (3rd) predicted class label.
		</pre>
	
	nearest_neighbors (dict):
		A dictionary where keys are protein IDs and values are lists of nearest neighbor information,
		including their t-SNE coordinates and PCA distances.
		<pre>
			"protein_ID": str, the identifier of the neighbor protein.
			"pca_distance": float, the PCA distance to the neighbor protein.
			"tsne_1": float, the first t-SNE coordinate of the neighbor protein.
			"tsne_2": float, the second t-SNE coordinate of the neighbor protein.
		</pre>

	Returns
	_______
	None. The function modifies the `user_data` list in place, adding estimated t-SNE coordinates and
	distances to the nearest neighbor for each protein record.
	"""
	for record in user_data:
		protein_id = record["protein_ID"]
		if protein_id in nearest_neighbors:
			nn_info = nearest_neighbors[protein_id]

			# Extract t-SNE coordinates and PCA distances of the nearest neighbors.
			neighbor_tsne_coords = [ [nn["tsne_1"], nn["tsne_2"]] for nn in nn_info ]
			pca_distances_list = [ nn["pca_distance"] for nn in nn_info ]

			# Compute weighted coordinates based on the distances to the nearest neighbors.
			weighted_coords = _weighted_coordinates(neighbor_tsne_coords, pca_distances_list)

			# Update the record with estimated t-SNE coordinates and distance to nearest neighbor.
			record["tsne_1"] = weighted_coords[0]
			record["tsne_2"] = weighted_coords[1]
