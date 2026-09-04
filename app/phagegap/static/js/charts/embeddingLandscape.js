import { Chart } from "./chart.js";
import { CATEGORY_COLORS, SUBCATEGORY_MAP, NA_COLOR } from "../constants.js";
import {
	metadataTable,
	userdataTable,
	nearestNeighbors,
	selectedId,
	nnId,
} from "../main.js";
import { displayNotification } from "../utility.js";

/**
 * The symbol size for user-submitted data points in the embedding landscape chart.
 */
const USER_DATA_SIZE = 10;

/**
 * Class to build and manage the embedding landscape chart using ECharts.
 * 
 * Extends the base {@link Chart} class to provide specific functionality for the embedding landscape chart,
 * including filling the chart with metadata, updating it with user-submitted data, and handling interactions
 * such as highlighting nearest neighbors.
 */
export class EmbeddingLandscape extends Chart {
	/**
	 * Constructor for the EmbeddingLandscape class.
	 *
	 * Sets an additional event listener for the `dataZoom` event to handle zooming and adjust symbol sizes accordingly.
	 *
	 * @param {element} element The DOM element to bind the chart to.
	 */
	constructor(element) {
		super(element);
		this.showReferenceIdentical = true;
		this.noReferenceIdentical = 0;
	}

	/**
	 * Fills the embedding landscape chart with data from the global `metadataTable`.
	 */
	fill() {
		// Ensure that the chart and metadata table are initialized before proceeding.
		if (!this.echart) return;
		if (!metadataTable) return;

		// Prepare the series data for each category based on the metadata table.
		var categorySeries = {};
		var metadataGroups = metadataTable.groups("category");
		for (const c of Object.keys(CATEGORY_COLORS)) {
			categorySeries[c] = {
				type: "scatter",
				name: c,
				symbolSize: 1,
				itemStyle: {
					color: CATEGORY_COLORS[c] || NA_COLOR,
				},
				data: metadataGroups.get(c).map((entry) => ({
					name: entry.protein_ID,
					value: [entry.tsne_1, entry.tsne_2],
				})),
				blendMode: "source-over",
				animation: false,
				large: true,
			};
		}

		// Style definitions to hide grid lines and axis labels for a cleaner look.
		const noGridLines = {
			axisLabel: {
				show: false,
			},
			axisLine: {
				show: true,
			},
			axisTick: {
				show: false,
			},
			splitLine: {
				show: false,
			},
		};

		// Style definitions for background elements like legend, toolbox, and tooltip to enhance visibility.
		const backgroundStyle = {
			backgroundColor: "rgba(247, 249, 252, 0.9)", // Light blue background for better visibility.
			borderRadius: 5,
		};

		const option = {
			grid: {
				containLabel: false,
				left: "0",
				top: 10,
				right: "0",
				bottom: "0",
				show: false,
			},
			xAxis: {
				name: "t-SNE 1",
				nameLocation: "center",
				nameGap: 10,
				...noGridLines,
			},
			yAxis: {
				name: "t-SNE 2",
				nameLocation: "center",
				nameGap: 10,
				...noGridLines,
			},
			animation: false, // Disable animation for better performance with large datasets.
			legend: {
				orient: "horizontal",
				bottom: 30,
				left: "center",
				textStyle: {
					fontSize: 10,
				},
				show: true,
				...backgroundStyle,
			},
			// dataZoom components are only used for drag actions.
			dataZoom: [
				{
					id: "embeddingLandscapeDragX",
					type: "inside",
					xAxisIndex: 0,
					zoomOnMouseWheel: false,
				},
				{
					id: "embeddingLandscapeDragY",
					type: "inside",
					yAxisIndex: 0,
					zoomOnMouseWheel: false,
				},
			],
			toolbox: {
				left: "1%",
				top: "top",
				feature: {
					dataZoom: {},
					myToggleReferenceIdentical: {
						show: true,
						title: `Hide Reference-identical`,
						icon: "path://M512 0C229.230933 0 0 229.230933 0 512s229.230933 512 512 512 512-229.230933 512-512S794.769067 0 512 0z m0 960C264.601067 960 64 759.398933 64 512S264.601067 64 512 64s448 200.601067 448 448-200.601067 448-448 448z m0-832c-211.764267 0-384 172.235733-384 384s172.235733 384 384 384c211.764267 0 384-172.235733 384-384s-172.235733-384-384-384z m0 640c-141.3856 0-256-114.6144-256-256s114.6144-256 256-256c141.3856 0 256 114.6144 256 256s-114.6144 256-256 256z",
						onclick: () => {
							this.#toggleShowReferenceIdentical();
						},
					},
				},
				padding: 5,
				...backgroundStyle,
			},
			tooltip: {
				trigger: "item",
				formatter: this.#formatTooltip,
				...backgroundStyle,
			},
			series: [...Object.values(categorySeries)],
		};

		// Set the prepared option object to the ECharts instance and activate the chart.
		this.echart.setOption(option);
		this.setActive();
	}

	/**
	 * Updates the embedding landscape chart with user-submitted data from the global `userdataTable`.
	 */
	update() {
		if (!this.echart) return;
		if (!userdataTable) return;

		// Get the current option object.
		var option = this.echart.getOption();

		// Delete any existing `user_data` scatter series from the current options.
		option.series = option.series.filter(
			(series) => series.name !== "User Data",
		);

		// Add the new `user_data` scatter series to the current options.
		let d = [];
		userdataTable.entries().forEach((entry) => {
			let refId = nearestNeighbors[entry.protein_ID][0].pca_distance < 0.001;
			if (refId) {
				this.noReferenceIdentical++;
			}
			d.push({
				name: entry.protein_ID,
				value: [entry.tsne_1, entry.tsne_2],
				referenceIdentical: refId,
				itemStyle: {
					color: CATEGORY_COLORS[SUBCATEGORY_MAP[entry.top1]] || NA_COLOR,
				},
			});
		});
		option.series.push({
			type: "scatter",
			name: "User Data",
			symbolSize: USER_DATA_SIZE,
			itemStyle: {
				borderColor: "black",
				borderWidth: 0.8,
				opacity: 1,
			},
			data: d,
			zlevel: 13, // Render the "User Data" series above the other series to ensure visibility.
			markLine: {
				symbol: ["none", "none"],
				data: [], // This will be populated dynamically when a user data point is clicked.
				zlevel: 12,
			},
		});
		option.toolbox[0].feature.myToggleReferenceIdentical.title = `Hide ${this.noReferenceIdentical} Reference-identical`;

		// Update the chart with the new options.
		this.echart.setOption(option);

		// Display user notification about successful loading of the embedding landscape chart.
		displayNotification(
			`Loaded ${userdataTable.entries().length} proteins into <code>Embedding Landscape</code>.`,
			"Success",
			"success",
		);
	}

	/**
	 * Highlights the nearest neighbor of a clicked "User Data" point in the embedding landscape chart by drawing a
	 * line connecting the two points.
	 *
	 * @param {object} clicked An entry from the `userdataTable` corresponding to the clicked data point in the embedding landscape chart.
	 */
	highlight(clicked) {
		// Construct data for nearest-neighbor connecting lines.
		var d = [];
		var isFirst = true;
		for (const nn of nearestNeighbors[clicked.protein_ID]) {
			var nnMetadata = metadataTable.entry(nn.protein_ID);
			d.push([
				{
					coord: [clicked.tsne_1, clicked.tsne_2],
				},
				{
					name: nn.protein_ID,
					coord: [nnMetadata.tsne_1, nnMetadata.tsne_2],
					value: nn.pca_distance,
					label: {
						show: false,
						backgroundColor: "rgba(238, 243, 251, 0.5)",
						borderRadius: 3,
						formatter: (params) => {
							return `${params.name} d{sub|PCA}=${params.value.toFixed(2)}`;
						},
						rich: {
							sub: {
								fontSize: 8,
								verticalAlign: "bottom", // Shifts text downward relative to baseline
								padding: [0, 0, -2, 0], // Fine-tune vertical position [top, right, bottom, left]
							},
						},
					},
					emphasis: {
						label: {
							show: true,
							formatter: (params) => {
								return `${params.name} d{sub|PCA}=${params.value.toFixed(2)}`;
							},
							rich: {
								sub: {
									fontSize: 9,
									verticalAlign: "bottom", // Shifts text downward relative to baseline
									padding: [0, 0, -2, 0], // Fine-tune vertical position [top, right, bottom, left]
								},
							},
						},
						lineStyle: {
							width: 2,
						},
					},
					lineStyle: {
						color: "#000000",
						type: isFirst ? "solid" : "dotted",
						width: 1,
					},
				},
			]);

			if (isFirst) isFirst = false;
		}

		// Update the "User Data" series option.
		const option = this.echart.getOption();
		const updatedUserDataSeries = option.series.map((series) => {
			if (series.name !== "User Data") {
				return series; // Do not change series that are not "User Data".
			} else {
				return {
					...series,
					// Reduce the opacity of unselected points to emphasize the selected protein and its nearest neighbors.
					data: series.data.map((point) => {
						const isSelected = point.name === clicked.protein_ID;
						return {
							...point,
							itemStyle: {
								...point.itemStyle,
								opacity: isSelected ? 1 : 0.33,
							},
						};
					}),
					// Updates the mark line data to show lines connecting the selected protein to its nearest neighbors.
					markLine: {
						...series.markLine,
						data: d,
					},
				};

			}
		});

		// Set the updated series back to the chart option and apply it to the ECharts instance.
		this.echart.setOption({
			series: updatedUserDataSeries,
		});

		// Highlight the first nearest neighbor (first in the list) by default.
		this.highlightNearestNeighbor(0);
	}

	/**
	 * Highlights the nearest neighbor at the specified index in the embedding landscape chart by showing its label and
	 * increasing its line width.
	 * 
	 * @param {number} nnIndex The index of the nearest neighbor to highlight in the mark line data. 
	 */
	highlightNearestNeighbor(nnIndex) {
		// Access mark line data.
		var option = this.echart
			.getOption();
		// Iterate through all mark lines in the "User Data" series.
		option.series[11].markLine.data.forEach((line, index) => {
			if (index == nnIndex) {
				line[1].label.show = true; // Show the label for the selected nearest neighbor.
				line[1].lineStyle.width = 2; // Increase line width for the selected nearest neighbor.
			} else {
				line[1].label.show = false; // Hide the label for all lines.
				line[1].lineStyle.width = 1; // Reset line width to default.
			}
		});
		// Highlight the nearest neighbor at the specified index by changing its line style to solid and increasing its width.
		this.echart.setOption(option);
	}

	/**
	 * Resets the highlight on the embedding landscape chart by restoring the opacity of all "User Data" points
	 * and removing any mark lines connecting the selected protein to its nearest neighbors.
	 */
	resetHighlight() {
		// Update the "User Data" series option.
		const option = this.echart.getOption();
		const updatedUserDataSeries = option.series.map((series) => {
			if (series.name !== "User Data") {
				return series; // Do not change series that are not "User Data".
			} else {
				return {
					...series,
					// Reduce the opacity of unselected points to emphasize the selected protein and its nearest neighbors.
					data: series.data.map((point) => {
						return {
							...point,
							itemStyle: {
								...point.itemStyle,
								opacity: 1,
							},
						};
					}),
					// Updates the mark line data to show lines connecting the selected protein to its nearest neighbors.
					markLine: {
						...series.markLine,
						data: [],
					},
				};

			}
		});

		// Set the updated series back to the chart option and apply it to the ECharts instance.
		this.echart.setOption({
			series: updatedUserDataSeries,
		});
	}

	/**
	 * Observes zoom events on the embedding landscape chart and adjusts the symbol size of the scatter points based on the zoom level.
	 * 
	 * Note: This method should be called after the chart has been initialized. It can not be called in the constructor because the
	 * ECharts instance may not be ready at that time.
	 */
	observeZoom() {
		this.echart.on("dataZoom", (params) => {
			// Check if the event parameters carries a batch property.
			if (params.batch) {

				// Catch 'dataZoom' events fired by the chart option's dataZoom components (only intended for drag actions).
				if (params.batch.some((batch) => batch.dataZoomId === "embeddingLandscapeDragX"
					|| batch.dataZoomId === "embeddingLandscapeDragY")) {
					return;
				}

				const maxRange = 300; // Hard coded based on initial chart range. Adjust as necessary for new data.
				const xRange = params.batch[0].endValue - params.batch[0].startValue; // Get the current x-axis range.
				const yRange = params.batch[1].endValue - params.batch[1].startValue; // Get the current y-axis range.
				const zoomLevel = (Math.min(xRange, yRange) / maxRange).toFixed(2); // Calculate the zoom level based on the smaller of the two ranges.
				var newSymbolSize;

				if (isNaN(zoomLevel)) {
					newSymbolSize = 1; // Default symbol size if zoom level is not a number.
				} else {
					newSymbolSize = Math.min(4, Math.round(1 / zoomLevel)); // Adjust symbol size inversely proportional to zoom level, capped at a maximum size of 6.
				}

				this.echart.setOption({
					series: this.echart.getOption().series.map((series) => ({
						...series,
						symbolSize: series.name == "User Data" ? USER_DATA_SIZE : newSymbolSize,
					})),
				});
			}
		});
	}

	/**
	 * Filters the user-submitted data points in the embedding landscape chart based on a minimum probability threshold.
	 * 
	 * @param {number} minProbability The minimum probability threshold (in percentage) for displaying user-submitted data points
	 * in the embedding landscape chart. 
	 */
	filterPredictionProbability(minProbability) {
		if (!this.echart) return;
		if (!userdataTable) return;

		const option = this.echart.getOption();
		if (!option.series[11]) return; // Ensure the "User Data" series exists before proceeding.
		option.series[11].data.forEach((point) => {
			const info = userdataTable.entry(point.name);
			const probability = (info["P(top1)"] * 100);
			point.symbolSize = probability >= minProbability ? USER_DATA_SIZE : 0;
		});
		this.echart.setOption(option);
	}

	/**
	 * Toggles the visibility of user-submitted data points that are identical to reference data in the embedding landscape chart.
	 * 
	 * When toggled, the method updates the symbol size of points that are identical to reference data based on the current state
	 * of `showReferenceIdentical`. If `showReferenceIdentical` is true, identical points will be displayed with their original
	 * symbol size; if false, they will be hidden (symbol size set to 0).
	 */
	#toggleShowReferenceIdentical() {
		this.showReferenceIdentical = !this.showReferenceIdentical;
		const option = this.echart.getOption();
		if (!option.series[11]) return; // Ensure the "User Data" series exists before proceeding.
		option.series[11].data.forEach((point) => {
			if (point.referenceIdentical) {
				point.symbolSize = this.showReferenceIdentical ? USER_DATA_SIZE : 0;
			}
		});
		option.toolbox[0].feature.myToggleReferenceIdentical.title =
			`${this.showReferenceIdentical ? "Hide" : "Show"} ${this.noReferenceIdentical} Reference-identical`;
		this.echart.setOption(option);
	}

	/**
	 * Formats the tooltip content for the embedding landscape chart based on the series and data point being hovered over.
	 *
	 * @param {object} params ECharts event parameters for the `tooltip` event.
	 * @returns {string|undefined} The formatted tooltip content as an HTML string, or undefined if the tooltip should not be displayed.
	 */
	#formatTooltip(params) {
		if (params.data === undefined) return;
		if (params.seriesName === "User Data") {
			if (!userdataTable) return;

			// Special handling for markLine tooltips to display nearest neighbor information.
			if (params.componentType === "markLine") {
				if (params.name !== nnId.getValue()) {
					return `<small>Click to set nearest neighbor.</small>`;
				} else {
					return; // Display no tooltip if the nearest neighbor is already selected.
				}
			}

			const info = userdataTable.entry(params.data.name);
			const nearestNeighbor = nearestNeighbors[params.data.name][0];
			var content = `<code>ID: ${params.data.name}</code><br>`;
			let probability = (info["P(top1)"] * 100).toFixed(2);
			content += `<table border=1 frame=void rules=rows>`;
			content += `<tr><td>Description:</td><td><p style="max-width: 300px; font-size: small;">${info.description}</p></td></tr>`;
			content += `<tr><td>Predicted Category:</td><td>${SUBCATEGORY_MAP[info.top1]} › ${info.top1} (${probability}%)</td></tr>`;
			content += `<tr><td>Nearest Neighbor:</td><td>${nearestNeighbor.protein_ID} d<sub>PCA</sub>=${nearestNeighbor.pca_distance.toFixed(2)}</td></tr>`;
			content += `</table>`;
			if (selectedId.getValue() !== params.data.name) {
				content += `<br><code>Click for details.</code>`;
			}
		} else {
			if (!metadataTable) return;
			const info = metadataTable.entry(params.data.name);
			var content = `<code>ID: ${params.data.name} | Locus Tag: ${info.locus_tag}</code><br>`;
			content += `<table border=1 frame=void rules=rows>`;
			content += `<tr><td>Organism:</td><td>${info.organism}</td></tr>`;
			content += `<tr><td>Category:</td><td>${info.category} › ${info.subcategory}</td></tr>`;
			content += `<tr><td>Product:</td><td>${info.product}</td></tr>`;
			content += `</table>`;
		}
		return content;
	}
}
