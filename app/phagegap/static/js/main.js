import * as util from "./utility.js";
import { Observable } from "./observable.js";
import { ArqueroTable } from "./table.js";
import { CATEGORY_COLORS, NA_COLOR } from "./constants.js";
import { EmbeddingLandscape } from "./charts/embeddingLandscape.js";
import { CategoryProbabilities } from "./charts/categoryProbabilities.js";
import { StructureView } from "./charts/structureView.js";
import { SequenceAlignment } from "./charts/sequenceAlignment.js";
import {
	GenomicContext,
	populateFeatures,
	featureTable,
} from "./charts/genomicContext.js";

/**
 * Global {@link ArqueroTable} instance for storing metadata.
 */
export var metadataTable = null;

/**
 * Global {@link ArqueroTable} instance for storing user-submitted data.
 */
export var userdataTable = null;

/**
 * Global {@link Object} to store nearest neighbor information user-submitted data.
 */
export var nearestNeighbors = {};

/**
 * Global {@link EmbeddingLandscape} instance for managing the embedding landscape chart.
 */
export var embeddingLandscape = null;

/**
 * Global {@link CategoryProbabilities} instance for managing the category probabilities chart.
 */
export var categoryProbabilities = null;

/**
 * Global {@link StructureView} instance for managing the structure view.
 */
export var structureView = null;

/**
 * Global {@link GenomicContext} instance for managing the genomic context chart.
 */
export var genomicContext = null;

/**
 * Global {@link SequenceAlignment} instance for managing the sequence alignment chart.
 */
export var sequenceAlignment = null;

/**
 * The ID of the currently selected protein in the embedding landscape chart.
 */
export var selectedId = new Observable(null);

/**
 * The ID of the currently selected nearest neighbor protein.
 */
export var nnId = new Observable(null);

/**
 * The version of the PhageGAP application.
 */
const VERSION = "1.0.0";

selectedId.onChange(handleSelectionChange); // Set up a listener to propagate click events when the selected protein changes.
nnId.onChange(handleNearestNeighborChange); // Set up a listener to propagate click events when the nearest neighbor protein changes.

/**
 * Main initialization function.
 */
export function initApp() {
	// Initialize all MetroUI components.
	Metro.init();

	// Populate the about page index.
	indexUsageTab();

	// Initialize charts.
	embeddingLandscape = new EmbeddingLandscape(
		document.getElementById("embedding-landscape-chart-container"),
	);
	embeddingLandscape.observeZoom(); // Set up zoom observation for the embedding landscape chart.
	categoryProbabilities = new CategoryProbabilities(
		document.getElementById("category-probabilities-chart-container"),
	);
	categoryProbabilities.state.setValue("Run Function Prediction or Reload Session to start.")

	sequenceAlignment = new SequenceAlignment(
		document.getElementById("sequence-alignment-chart-container"),
	);

	genomicContext = new GenomicContext(
		document.getElementById("genomic-context-chart-container"),
	);
	genomicContext.state.setValue(
		"Submit a GFF file via Add Genomic Context in the toolbar.",
	);

	structureView = new StructureView(
		document.getElementById("structure-view-container"),
	);

	// Initialize selection propagation.
	Metro.getPlugin("#protein-select", "select").options.onChange = function (
		value,
	) {
		selectedId.setValue(value[0]); // Update the selected protein ID when a new option is selected from the dropdown.
	};
	Metro.getPlugin("#protein-select", "select").options.onClear = function () {
		selectedId.setValue(null); // Clear the selected protein ID when the selection is cleared from the dropdown.
	};

	// Initialize click handler for the embedding landscape and genomic context chart.
	embeddingLandscape.echart.on("click", (params) => {
		if (params.seriesName === "User Data") {
			if (params.componentType === "series") {
				// Update the selected protein when a "User Data" point is clicked.
				// The observer triggers the propagation of the click event to the appropriate handlers.
				Metro.getPlugin("#protein-select", "select").val(params.data.name);
			} else if (params.componentType === "markLine") {
				// Update the nearest neighbor protein when a nearest neighbor line is clicked.
				nnId.setValue(params.data.name);
			} else {
				return; // Ignore clicks on other components of the chart.
			}
		}
	});
	genomicContext.echart.on("click", (params) => {
		let protein_id = params.data.name;
		// Check if the clicked protein ID exists in the user-submitted data table before updating the selected protein.
		if (userdataTable && userdataTable.entry(protein_id) !== null) {
			Metro.getPlugin("#protein-select", "select").val(protein_id);
		}
	});

	// Bind functions to (toolbar) buttons.
	document.getElementById("input-submit-button").onclick = requestPrediction;
	document.getElementById("session-reload-button").onclick = restoreSession;
	document.getElementById("toolbar-genomic-context-button").onclick =
		requestFeatures;
	/*document.getElementById("toolbar-maximize-structure-button").onclick =
		toggleMaximizeStructureView;*/
	document.getElementById("toolbar-screenshot-button").onclick =
		captureScreenshot;
	document.getElementById("toolbar-download-results-button").onclick =
		downloadResults;
	document.getElementById("toolbar-download-session-button").onclick =
		downloadSession;
	
	// Promise for the presence of the `phagegap_consent` cookie.
	util.cookieDisclaimer().then(() => {
		// Enable the "Function Classification" tab in the navigation bar.
		enableClassificationTab();

		// Request metadata and initialize the metadata table.
		requestMetadata().then(() => {
			// Initialize the protein landscape chart after the metadata table has been loaded.
			embeddingLandscape.fill();
		});
	});
}

/**
 * Adds clickable links to the about page navigation index.
 *
 * For each `about-section` element of the about page, it retrieves the `id` attribute and creates 
 * a list item with a link that points to that section.
 * 
 * If the heading child element of the section is `h1`, no indentation is applied to the link. For all
 * other heading levels, an indentation is added to visually distinguish them in the navigation index.
 */
function indexUsageTab() {
	var usageNavIndex = $("#usage-nav-index");
	$(".usage-section").each(function () {
		var sectionId = $(this).attr("id");
		var heading = $(this).find("h1, h2, h3, h4, h5, h6").first();
		var linkText = heading.text();
		var indent = heading.is("h1") ? "" : "class='ml-4 text-light'";
		usageNavIndex.append(
			`<li><a href="#${sectionId}" ${indent}>${linkText}</a></li>`,
		);
	});
}

/**
 * Enables the "Function Classification" tab in the navigation bar, allowing users to access the classification functionality.
 */
function enableClassificationTab() {
	// Enable the classification tab in the navigation bar.
	var classificationTab = document.getElementById("classification-tab");
	classificationTab.removeAttribute("disabled");
}

/**
 * Handles changes in the selected protein ID, updating the relevant charts and UI elements accordingly.
 */
function handleSelectionChange() {
	if (!userdataTable) return;
	if (selectedId.getValue() !== null) {
		const selectedProteinEntry = userdataTable.entry(selectedId.getValue());

		// Show the category probabilities for the clicked protein in the category probabilities chart.
		if (categoryProbabilities && selectedProteinEntry)
			categoryProbabilities.show(selectedProteinEntry);
		// Clear the loading state of the category probabilities chart.
		categoryProbabilities.state.setValue(null);

		// Highlight the nearest neighbor of the clicked "User Data" point in the embedding landscape chart.
		if (embeddingLandscape && selectedProteinEntry)
			embeddingLandscape.highlight(selectedProteinEntry);

		// Zoom into the genomic context feature corresponding to the clicked point, if it exists.
		if (genomicContext && selectedProteinEntry) genomicContext.zoom();

		// Set the nearest neighbor information to the first nearest neighbor of the clicked protein.
		nnId.setValue(nearestNeighbors[selectedProteinEntry.protein_ID][0].protein_ID);

		// Add information about the selected protein to the selection info container.
		document.getElementById("selection-clear-button").style.display =
			"inline-block";
		document.getElementById("selection-info-label").innerHTML =
			`Selected <code>${selectedId.getValue()}</code>`;
	} else {
		// If the selection is null (cleared), hide and clear respective UI elements and reset zoom.
		categoryProbabilities.clear();
		embeddingLandscape.resetHighlight();
		genomicContext.zoom();
		structureView.clear();
		sequenceAlignment.clear();
		document.getElementById("nearest-neighbor-cloak").style.display = "block";
		document.getElementById("nearest-neighbor-cloak").innerHTML = `
		<p class="reduce-2 text-center mt-8">
			Click a User Data point in the embedding landscape to view nearest neighbor information.
		</p>
		`;
		document.getElementById("nearest-neighbor-id-tag").innerHTML = "";
		document.getElementById("nearest-neighbor-class-tag").innerHTML = "";

		// Reset information about the selected protein in the selection info container.
		document.getElementById("selection-clear-button").style.display = "none";
		document.getElementById("selection-info-label").innerHTML =
			`No protein selected.`;
	}
}

/**
 * Handles changes in the nearest neighbor protein ID, requesting structure information from the server and updating
 * the structure view and sequence alignment chart accordingly.
 * 
 * Note: In principle, any valid protein ID can be used to request structure information, but in practice,
 * this function is intended to be used with the protein ID of one of the reported nearest neighbors of the
 * currently selected user-submitted data point.
 */
function handleNearestNeighborChange() {
	if (!userdataTable) return;
	if (selectedId.getValue() == null) return;
	if (nnId.getValue() == null) return;

	// Extract current selection data.
	const currentSelection = userdataTable.entry(selectedId.getValue());
	let nnProteinId = nnId.getValue();
	let nnIndex = nearestNeighbors[currentSelection.protein_ID].findIndex((nn) => nn.protein_ID === nnProteinId);
	
	// Highlight the nearest neighbor in the embedding landscape chart.
	embeddingLandscape.highlightNearestNeighbor(nnIndex);

	// Request structure data of the nearest neighbor protein from the server and update the structure view and alignment chart accordingly.
	var formData = new FormData();
	formData.append("protein_seq", currentSelection.sequence);
	axios
		.post(`/api/classifier/nninfo`, formData, {
			headers: {
				"X-CSRF-Token": document.querySelector('meta[name="csrf-token"]')
					.content,
			},
			params: {
				nn_protein_id: nnProteinId,
			},
		})
		.then((response) => {
			// Fade out the nearest neighbor information overlay to reveal the structure view and sequence alignment chart.
			document.getElementById("nearest-neighbor-cloak").style.display = "none";

			// Update the nearest neighbor tags.
			document.getElementById("nearest-neighbor-id-tag").innerHTML = `${nnProteinId}`;
			var nnMetadata = metadataTable.entry(nnProteinId);
			console.log(nnMetadata);
			var bgClr = CATEGORY_COLORS[nnMetadata.category] || NA_COLOR;
			document.getElementById("nearest-neighbor-class-tag").innerHTML = `${nnMetadata.subcategory}`;
			document.getElementById("nearest-neighbor-class-tag").style.backgroundColor = bgClr;

			// Fill the sequence alignment chart with returned alignment data.
			sequenceAlignment.fill(
				response.data.sequence_alignment[0], // Sequence of the nearest neighbor protein.
				response.data.sequence_alignment[1], // Sequence of the selected protein.
				nnProteinId,
				currentSelection.protein_ID,
			);

			// Provide the sequence identity information in the sequence alignment chart.
			sequenceAlignment.setIdentity(response.data.sequence_identity);

			// Set structure and information label of the structure view.
			structureView.clear(); // Clear the structure view before adding new content.
			structureView.fill(response.data.structure_data, "cif"); // Add the protein structure to the structure view.
			structureView.setInfo(
				response.data.structure_plddt_mean,
				response.data.structure_ptm,
			);
		})
		.catch((error) => {
			// Display an error notification as state of the sequence alignment chart.
			document.getElementById("nearest-neighbor-cloak").style.display = "block";
			document.getElementById("nearest-neighbor-cloak").innerHTML = `
			<i class="fa-solid fa-triangle-exclamation" style="color: #cb0e40;"></i>
			<p class="reduce-2 text-center mt-8">
				Failed to retrieve structure and sequence data of the selected nearest-neighbor. Please try another.
			</p>
			`;

			console.error(error);
			let message = error.message;
			if (error.response.data) {
				message += `<hr>${error.response.data}`;
			}
			util.displayNotification(
				"Failed to retrieve nearest neighbor information: " + message,
				"Error",
				"warning",
			);
		});
}

/**
 * Captures a screenshot of the current state of the application, including the embedding landscape chart, structure view,
 * and other visible elements, and prompts the user to download it as a PNG image.
 * 
 * The screenshot is captured using the `html2canvas` library, which renders the specified DOM element into a canvas.
 * The resulting image is then converted to a data URL and downloaded as a PNG file.
 */
function captureScreenshot() {
	try {
		html2canvas(document.getElementById("app-area"), {
			scale: 4,
		}).then((canvas) => {
			// Convert the canvas to a data URL and create a download link.
			const dataURL = canvas.toDataURL("image/png");
			const downloadLink = document.createElement("a");
			downloadLink.href = dataURL;
			downloadLink.download = `phagegap-screenshot-${new Date().toISOString().slice(0, 10)}.png`;
			downloadLink.click(); // Trigger the download.
			downloadLink.remove(); // Clean up the download link element after use.
		});
	} catch (error) {
		console.error(error);
		util.displayNotification(
			"Failed to capture screenshot: " + error.message,
			"Error",
			"warning",
		);
	}
}

/**
 * Returns a promise that resolves when the user-submitted data table has been initialized with the provided data.
 * 
 * @param {any} data See https://idl.uw.edu/arquero/api/#from for the expected data format.
 * @returns {Promise} A promise that resolves when the user-submitted data table has been successfully initialized. 
 */
function promiseUserdataTable(data) {
	return new Promise((resolve, reject) => {
		try {
			// Build new table if no data is present, otherwise update the existing table with new data.
			if (!userdataTable) {
				userdataTable = new ArqueroTable(data);
			} else {
				userdataTable.insert(data);
			}
			resolve();
		} catch (error) {
			reject(error);
		}
	});
}

/**
 * Asynchronously fetches metadata from the server and initializes the metadata table for use in the application.
 * 
 * @returns a {@link Promise} that resolves when the metadata has been successfully loaded and the table has been initialized.
 */
function requestMetadata() {
	return axios
		.get("/api/metadata", {
			headers: {
				"X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content
			},
		})
		.then((response) => {
			// Initialize the metadata table with the received data.
			metadataTable = new ArqueroTable(response.data);
		})
		.catch((error) => {
			console.error(error);
			let message = error.message;
			if (error.response.data) {
				message += `<hr>${error.response.data}`;
			}
			util.displayNotification(
				"Failed to fetch metadata: " + message,
				"Error",
				"warning",
			);
		});
}

/**
 * Sends a classification request to the server with the provided input data.
 */
function requestPrediction() {
	document.getElementById("app-area").style.display = "block"; // Ensure the app area is visible before proceeding.

	try {
		// Check if a file has been selected for upload.
		var fileInput = document.getElementById("data-input-file");
		var formData = new FormData();
		if (fileInput.files.length > 0) {
			// Check if the file input has a file selected.
			formData.append("file", fileInput.files[0]);
		}
		var textInput = document.getElementById("data-input-text");
		var textData = textInput.value.trim();
		if (textData) {
			// Check if the text input has a value.
			formData.append("text", textData);
		}
		// Check if the form data is empty (no file or text input provided).
		if (formData.entries().next().done) {
			util.displayNotification(
				"No input data provided. Please select a file or enter text for classification.",
				"Info",
				"alert",
			);
			return false;
		}
	} catch (error) {
		console.error(error);
		util.displayNotification(
			"Failed to prepare input data for classification: " + error.message,
			"Error",
			"warning",
		);
		return false;
	}

	// Show notification that the classification request is being processed.
	util.displayNotification(
		"Submitted data for classification. This may take a few moments...",
		"Info",
		"info",
	);

	// Send the form data to the server using a POST request.
	axios
		.post("/api/classifier/predict", formData, {
			headers: {
				"X-CSRF-Token": document.querySelector('meta[name="csrf-token"]')
					.content,
			},
		})
		.then((response) => {
			// Update the global nearest neighbor information with the received data.
			Object.assign(nearestNeighbors, response.data.nearest_neighbors);

			// Process the user-submitted data received from the server and update the application state accordingly.
			processUserData(response.data.predictions);
		})
		.catch((error) => {
			console.error(error);
			let message = error.message;
			if (error.response.data) {
				message += `<hr>${error.response.data}`;
			}
			util.displayNotification(
				"Failed to classify submitted data: " + message,
				"Error",
				"warning",
			);
		});
}

/**
 * Processes classification results into the application.
 * 
 * This is either the result of a classification request or a session restore. It initializes the user-data table,
 * updates the embedding landscape chart, and fills the genomic context chart with the received data.
 * 
 * @param {Array} data The user-submitted data to be processed, received from the server after a classification request or session restore. 
 */
function processUserData(data) {
	// Collapse the data input section.
	const collapse = Metro.getPlugin("#data-input", "collapse");
	if (collapse && !collapse.options.collapsed) {
		collapse.collapse();
	}

	// Initialize the user-data table with the received data.
	promiseUserdataTable(data).then(() => {
		// Update the selection search dropdown with protein IDs from the user-submitted data table.
		const proteinSelectPlugin = Metro.getPlugin("#protein-select", "select");
		proteinSelectPlugin.reset(); // Clear existing options in the selection dropdown.
		var proteinIds = userdataTable.table.array("protein_ID");
		proteinIds.forEach((id) => {
			proteinSelectPlugin.addOption(id, id, false);
		});
		proteinSelectPlugin.clear();
		
		// Update the embedding landscape chart to reflect the newly loaded user-submitted data.
		embeddingLandscape.update();

		// Update the genomic context chart to reflect the newly loaded user-submitted data, if applicable.
		genomicContext.fill();

		categoryProbabilities.state.setValue(
			"Click on a User Data point to view function predictions.",
		);
	});
}

/**
 * Handles initialization and updating of the genomic context chart with data from a GFF file.
 * 
 * Prompts the user to select a GFF file, sends the file to the server for processing, and updates the genomic context chart with the received data.
 */
function requestFeatures() {
	// Show file input dialog to select a GFF file for genome context.
	util.promptFile(".gff,.gff3")
		.then((formData) => {
			// Send the form data to the server using a POST request.
			axios
				.post("/api/gff", formData, {
					headers: {
						"X-CSRF-Token": document.querySelector('meta[name="csrf-token"]')
							.content,
					},
				})
				.then((response) => {
					processFeatures(response.data);
				})
				.catch((error) => {
					console.error(error);
					util.displayNotification(
						"Failed to load genome context: " + error.message,
						"Error",
						"warning",
					);
				});
		})
		.catch((error) => {
			console.error(error);
			let message = error.message;
			if (error.response.data) {
				message += `<hr>${error.response.data}`;
			}
			util.displayNotification(
				"Failed to load genome context: " + message,
				"Error",
				"warning",
			);
		});
}

/**
 * Processes feature data into the application.
 * 
 * This is either the result of a GFF file upload or a session restore. It initializes the features table and updates the genomic
 * context chart with the received data.
 * 
 * @param {Array} data The feature data to be processed, received from the server after a GFF file upload or session restore. 
 */
function processFeatures(data) {
	populateFeatures(data).then(() => {
		// Update the genome context chart with the received data.
		genomicContext.fill();
	});
}

/**
 * Restores a previously saved session by prompting the user to select a compressed JSON file containing session data.
 * 
 * The session data includes selected protein, user-submitted data, and features. The function decompresses the file,
 * parses the JSON content, and updates the application state accordingly.
 */
async function restoreSession() {
	try {
		const formData = await util.promptFile(".gz,.json.gz");
		const file = formData.get("file");

		const compressed = await file.arrayBuffer();
		const content = await util.decompress(compressed, "gzip");

		const sessionData = JSON.parse(content);

		new Promise((resolve, reject) => {
			// Load features.
			if (typeof sessionData.features == 'string') {
				sessionData.features = JSON.parse(sessionData.features);
				processFeatures(sessionData.features);
			}

			// Load user-submitted data.
			if (typeof sessionData.userdata == "string") {
				sessionData.userdata = JSON.parse(sessionData.userdata);
				processUserData(sessionData.userdata);
			}

			// Load nearest neighbor information.
			if (typeof sessionData.nearestNeighbors == "object") {
				Object.assign(nearestNeighbors, sessionData.nearestNeighbors);
			}

			resolve();
		}).then(() => {
			// Check if the value of selected in the session data is valid and exists in the user-submitted data table.
			if (
				sessionData.selectedUserData &&
				userdataTable &&
				userdataTable.entry(sessionData.selectedUserData) !== null
			) {
				Metro.getPlugin("#protein-select", "select").val(sessionData.selectedUserData);
			} else {
				categoryProbabilities.state.setValue(
					"Click on a User Data point to view function predictions.",
				);
			}

			// Check if the value of nearest neighbor in the session data is valid and exists in the user-submitted data table.
			if (
				sessionData.selectedNearestNeighbor &&
				nearestNeighbors &&
				metadataTable &&
				metadataTable.entry(sessionData.selectedNearestNeighbor) !== null
			) {
				nnId.setValue(sessionData.selectedNearestNeighbor);
			}
		});
	} catch (error) {
		console.error(error);
		util.displayNotification(
			"Failed to restore session: " + error.message,
			"Error",
			"warning",
		);
	}
}

/**
 * Downloads the user-submitted data table as a TSV file, allowing users to save their classification results locally.
 * 
 * The downloaded file is named with the current date in the format `phagegap-results-YYYY-MM-DD.tsv`.
 */
function downloadResults() {
	if (!userdataTable) {
		util.displayNotification(
			"No session data available to download. Please submit data first.",
			"Info",
			"alert",
		);
		return;
	}
	util.downloadBlob(
		new Blob([userdataTable.table.toCSV({ delimiter: "\t" })], {
			type: "text/plain",
		}),
		`phagegap-results-${new Date().toISOString().slice(0, 10)}.tsv`,
	);
}

/**
 * Downloads the current session data, including selected protein, user-submitted data, and features, as a compressed JSON file.
 * 
 * The downloaded file is named with the current date in the format `phagegap-session-YYYY-MM-DD`.
 */
function downloadSession() {
	if (!userdataTable) {
		util.displayNotification(
			"No session data available to download. Please submit data first.",
			"Info",
			"alert",
		);
		return;
	}
	let userdataRecords = userdataTable.table.toJSON({ type: "rows" });
	let featureRecords = [];
	if (featureTable) {
		featureRecords = featureTable.table.toJSON({ type: "rows" });
	}
	// Create a JSON object containing the session data.
	const sessionJSON = JSON.stringify(
		{
			selectedUserData: selectedId.getValue(),
			selectedNearestNeighbor: nnId.getValue(),
			userdata: userdataRecords,
			nearestNeighbors: nearestNeighbors,
			features: featureRecords,
			time: new Date().toISOString(),
			version: VERSION,
		},
		null,
	);
	util.compress(sessionJSON, "gzip").then((compressed) => {
		util.downloadBlob(
			new Blob([compressed], { type: "application/gzip" }),
			`phagegap-session-${new Date().toISOString().slice(0, 10)}.json.gz`,
		);
	});
}
