import * as util from "./utility.js";
import { Observable } from "./observable.js";
import { ArqueroTable } from "./table.js";
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
export var selected = new Observable(null);

/**
 * The version of the PhageGAP application.
 */
const VERSION = "1.0.0";

selected.onChange(handleSelectionChange); // Set up a listener to propagate click events when the selected protein changes.

/**
 * Main initialization function.
 */
export function initApp() {
	// Initialize all MetroUI components.
	Metro.init();

	// Populate the about page index.
	indexAbout();

	// Initialize charts.
	embeddingLandscape = new EmbeddingLandscape(
		document.getElementById("embedding-landscape-chart-container"),
	);
	embeddingLandscape.observeZoom(); // Set up zoom observation for the embedding landscape chart.
	categoryProbabilities = new CategoryProbabilities(
		document.getElementById("category-probabilities-chart-container"),
	);
	sequenceAlignment = new SequenceAlignment(
		document.getElementById("sequence-alignment-chart-container"),
	);
	sequenceAlignment.observeZoom(); // Set up zoom observation for the sequence alignment chart.
	structureView = new StructureView(
		document.getElementById("structure-view-container"),
	);
	genomicContext = new GenomicContext(
		document.getElementById("genomic-context-chart-container"),
	);

	// Initialize click handler for the embedding landscape chart.
	embeddingLandscape.echart.on("click", (params) => {
		if (params.seriesName === "User Data") {
			// Update the selected protein when a "User Data" point is clicked. The observer triggers the propagation of the click event to the appropriate handlers.
			selected.setValue(params.data.name);
		}
	});

	// Bind functions to (toolbar) buttons.
	document.getElementById("input-submit-button").onclick = requestPrediction;
	document.getElementById("session-reload-button").onclick = restoreSession;
	document.getElementById("toolbar-genomic-context-button").onclick =
		requestFeatures;
	document.getElementById("toolbar-maximize-structure-button").onclick =
		toggleMaximizeStructureView;
	document.getElementById("toolbar-screenshot-button").onclick =
		captureScreenshot;
	document.getElementById("toolbar-download-results-button").onclick =
		downloadResults;
	document.getElementById("toolbar-download-session-button").onclick =
		downloadSession;
	document.getElementById("selection-clear-button").onclick = () => {
		selected.setValue(null);
	};

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
function indexAbout() {
	var aboutNavIndex = $("#about-nav-index");
	$(".about-section").each(function () {
		var sectionId = $(this).attr("id");
		var heading = $(this).find("h1, h2, h3, h4, h5, h6").first();
		var linkText = heading.text();
		var indent = heading.is("h1") ? "" : "class='ml-4 text-light'";
		aboutNavIndex.append(
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
 * Propagates a click event on user-submitted data from the embedding landscape chart to the appropriate handlers.
 */
function handleSelectionChange() {
	if (selected.getValue() !== null) {
		const clicked = userdataTable.entry(selected.getValue());

		// Show the category probabilities for the clicked protein in the category probabilities chart.
		if (categoryProbabilities && clicked) categoryProbabilities.show(clicked);

		// Highlight the nearest neighbor of the clicked "User Data" point in the embedding landscape chart.
		if (embeddingLandscape && clicked) embeddingLandscape.highlight(clicked);

		// Zoom into the genomic context feature corresponding to the clicked point, if it exists.
		if (genomicContext && clicked) genomicContext.zoom();

		// Request structure data of the nearest neighbor protein from the server and update the structure view and alignment chart accordingly.
		var formData = new FormData();
		formData.append("protein_seq", clicked.sequence);
		axios
			.post(`/api/classifier/nninfo`, formData, {
				headers: {
					"X-CSRF-Token": document.querySelector('meta[name="csrf-token"]')
						.content,
				},
				params: {
					nn_protein_id: clicked.nearest_neighbor_ID,
				},
			})
			.then((response) => {
				sequenceAlignment.fill(
					response.data.sequence_alignment[0],
					response.data.sequence_alignment[1],
					clicked.nearest_neighbor_ID,
					selected.getValue(),
				);

				structureView.clear(); // Clear the structure view before adding new content.
				structureView.fill(response.data.structure_data, "cif"); // Add the protein structure to the structure view.
				structureView.setInfo(
					clicked.nearest_neighbor_ID,
					response.data.structure_plddt_mean,
					response.data.structure_ptm,
				);
			})
			.catch((error) => {
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
		
		// Toggle display of the selection info container in the toolbar to show the currently selected protein.
		document.getElementById("selection-info-container").style.display = "block";
		document.getElementById("selection-info-label").innerHTML = `Selected <code>${selected.getValue()}</code>`;
	} else {
		// If the selection is null (cleared), hide and clear respective UI elements and reset zoom.
		categoryProbabilities.clear();
		embeddingLandscape.resetHighlight();
		genomicContext.zoom();
		structureView.clear();
		sequenceAlignment.clear();
		document.getElementById("selection-info-container").style.display = "none";
		document.getElementById("selection-info-label").innerHTML = "";
	}
}

/**
 * Toggles the maximization state of the structure view in the user interface, adjusting the layout of the embedding landscape
 * chart and structure display accordingly.
 * 
 * - When maximized, the structure view occupies more space, and the category probabilities chart is hidden.
 * - When minimized, the structure view is reduced in size, and the category probabilities chart is displayed.
 */
function toggleMaximizeStructureView() {
	let structureViewLabel = document.getElementById("structure-view-label");
	if (structureViewLabel.hasAttribute("minimized")) {
		// Maximize the structure view element.

		// Remove indicator attribute.
		structureViewLabel.removeAttribute("minimized");

		// Switch label.
		structureViewLabel.innerHTML = "Nearest Neighbor Structure";

		// Switch button label.
		document.getElementById("toolbar-maximize-structure-button").innerHTML =
			`<i class="fa-solid fa-picture-in-picture"></i> Minimize Structure View`;

		// Adjust the layout of the embedding landscape chart and structure display cells to accommodate the maximized structure view.
		document
			.getElementById("embedding-landscape-chart-cell")
			.classList.remove("cell-9");
		document
			.getElementById("embedding-landscape-chart-cell")
			.classList.add("cell-6");
		document
			.getElementById("structure-view-cell")
			.classList.remove("cell-3");
		document.getElementById("structure-view-cell").classList.add("cell-6");

		// Adjust the container widths of the embedding landscape chart and structure display to fill the available space.
		document.getElementById("embedding-landscape-chart-container").style.width =
			"48vw";
		document.getElementById("structure-view-container").style.width = "50vw";
		document.getElementById("structure-view-container").style.height = "60vh";

		// Hide the category probabilities chart when the structure view is maximized.
		document.getElementById(
			"category-probabilities-chart-container",
		).style.display = "none";
	} else {
		// Minimize the structure view element.

		// Add indicator attribute.
		structureViewLabel.setAttribute("minimized", "");

		// Switch label.
		structureViewLabel.innerHTML =
			"Function Prediction and Nearest Neighbor Structure";

		// Switch button label.
		document.getElementById("toolbar-maximize-structure-button").innerHTML =
			`<i class="fa-solid fa-picture-in-picture"></i> Maximize Structure View`;

		// Adjust the layout of the embedding landscape chart and structure display cells to accommodate the minimized structure view.
		document
			.getElementById("embedding-landscape-chart-cell")
			.classList.remove("cell-6");
		document
			.getElementById("embedding-landscape-chart-cell")
			.classList.add("cell-9");
		document
			.getElementById("structure-view-cell")
			.classList.remove("cell-6");
		document.getElementById("structure-view-cell").classList.add("cell-3");

		// Adjust the container widths of the embedding landscape chart and structure display to fill the available space.
		document.getElementById("embedding-landscape-chart-container").style.width =
			"73vw";
		document.getElementById("structure-view-container").style.width = "27vw";
		document.getElementById("structure-view-container").style.height =
			"40vh";

		// Show the category probabilities chart when the structure view is minimized.
		document.getElementById(
			"category-probabilities-chart-container",
		).style.display = "block";
	}
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
			processUserData(response.data);
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
		// Update the embedding landscape chart to reflect the newly loaded user-submitted data.
		embeddingLandscape.update();

		// Update the genomic context chart to reflect the newly loaded user-submitted data, if applicable.
		genomicContext.fill();
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
		if (typeof sessionData.features == 'string') {
			sessionData.features = JSON.parse(sessionData.features);
			processUserData(sessionData.userdata);
		}
		if (typeof sessionData.userdata == 'string') {
			sessionData.userdata = JSON.parse(sessionData.userdata);
			processFeatures(sessionData.features);
		}

		if (sessionData.selected != null) selected.setValue(sessionData.selected);
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
			selected: selected.getValue(),
			userdata: userdataRecords,
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
