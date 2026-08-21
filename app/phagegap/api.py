"""
API endpoints for the `phagegap` Flask app.
"""

from __future__ import annotations

import hmac
import secrets
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


# Initialize Flask-Limiter for rate limiting API requests.
limiter = Limiter(key_func=get_remote_address, app=app)


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
@limiter.limit("10 per minute")  # Limit to 50 requests per minute per IP address.
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
@limiter.limit("60 per minute")  # Limit to 50 requests per minute per IP address.
def serve_nearest_neighbor_information():
	"""Serves information about the nearest neighbor of a user-provided protein.

	Expects the following parameters:
	- `nn_protein_id`: The protein ID of the nearest neighbor as argument in the query string.
	- `protein_seq`: The amino acid sequence of the user-provided protein as form data.

	This endpoint retrieves the structure of the nearest neighbor protein from the PhageGap classifier service,
	extracts its amino acid sequence, and aligns it with the user-provided sequence. The response includes the
	structure data and the alignment information.

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

		if app.extensions["emulate"]:
			# Emulate the behavior of the PhageGap classifier service for testing purposes.
			cif_content = Path(app.static_folder).joinpath("resources/data/emulate_nn_structure.cif").read_text(encoding="utf-8")
			response_dict = {"structure_data": cif_content, "structure_plddt_mean": 0.42, "structure_ptm": 0.42}
		else:
			# Request the structure of the nearest neighbor protein from the PhageGap classifier service.
			response = requests.get(
				f"{app.extensions['classifier_url']}/structure", {"protein_id": nn_protein_id},
				headers={
					"Authorization": (
						f"Bearer {app.extensions['api_token']}"
					),
					"Accept": "application/json",
				},
				timeout=120,
			)
			response.raise_for_status()
			response_dict = response.json()

		# Extract the amino acid sequence from the CIF content.
		cif_content = response_dict.get("structure_data")
		if cif_content is None:
			return f"No structure data found for {nn_protein_id}.", 404
		nn_seq = read_sequence_from_cif(cif_content)

		# Align the nearest neighbor sequence with the sequence provided by the user.
		response_dict["sequence_alignment"] = list(align_sequences(nn_seq, protein_seq))

		return response_dict, 200
	except Exception as e:
		print_exc()
		return f"Failed to retrieve nearest neighbor information: {str(e)}", 500


@app.route("/api/gff", methods=["POST"])
@limiter.limit("2 per minute")  # Limit to 50 requests per minute per IP address.
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
@limiter.limit("1 per minute")  # Limit to 50 requests per minute per IP address.
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

		if len(sequence_records) == 0:
			return "No valid FASTA records found.", 400

		# Prepare user data structure to store results.
		# Note: It is important to duplicate the protein ID as key for later merging.
		user_data = {
			"records": {
				record.id: {
					"protein_ID": record.id,
					"description": record.description,
					"sequence": str(record.seq)
				}
				for record in sequence_records
			},
		}

		''' TODO: Enable this once we have a working SocketIO connection to the client.
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
			# Emulate the behavior of the PhageGap classifier service for testing purposes.
			for record in user_data["records"].values():
				# Emulate classification results.
				record["top1"] = "Terra"
				record["P(top1)"] = round(np.random.uniform(0.7, 1.0), 3)
				record["top2"] = "incognita"
				record["P(top2)"] = round(np.random.uniform(0.5, 0.7), 3)
				record["top3"] = "manet"
				record["P(top3)"] = round(np.random.uniform(0.0, 0.5), 3)

				# Emulate t-SNE coordinates by randomly selecting two entries from the metadata and using their t-SNE coordinates.
				sample = app.extensions["metadata"].sample(n=2)
				record["tsne_1"] = sample.iloc[0]["tsne_1"]
				record["tsne_2"] = sample.iloc[1]["tsne_2"]
				record["nearest_neighbor_ID"] = sample.iloc[0]["protein_ID"]
				record["nearest_neighbor_distance"] = 42
		else :
			# Forward the original FASTA text to the classifier apptainer.
			response = requests.post(
				f"{app.extensions['classifier_url']}/predict",
				headers={
					"Authorization": (
						f"Bearer {app.extensions['api_token']}"
					),
					"Content-Type": "text/x-fasta; charset=utf-8",
					"Accept": "application/json",
				},
				data=sequence_text.encode("utf-8"),
				timeout=1800,
			)
			response.raise_for_status()
			response_data = response.json()

			# Merge the results from the classifier with the user data.
			for record in response_data:
				protein_id = record.get("protein_ID")
				if protein_id in user_data["records"]:
					# TODO: Keys need to be adjusted, if changed in classifier.
					for key in ["top1", "P(top1)", "top2", "P(top2)", "top3", "P(top3)", "nearest_neighbor_ID", "nearest_neighbor_distance", "tsne_1", "tsne_2"]:
						user_data["records"][protein_id][key] = record.get(key)

		# Return the merged results to the client.
		return list(user_data["records"].values()), 200

	except Exception as e:
		print_exc()
		return f"Classification request failed: {str(e)}", 500
