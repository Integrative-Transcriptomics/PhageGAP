import { Chart } from "./chart.js";
import { CATEGORY_COLORS, SUBCATEGORY_MAP, NA_COLOR } from "../constants.js";
import { metadataTable, userdataTable, nearestNeighbors } from "../main.js";
import { displayNotification } from "../utility.js";

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
		for (const [category, entries] of metadataTable.groups("category")) {
			categorySeries[category] = {
				type: "scatter",
				name: category,
				symbolSize: 1,
				itemStyle: {
					color: CATEGORY_COLORS[category] || NA_COLOR,
				},
				data: entries.map((entry) => ({
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
			toolbox: {
				left: "left",
				top: "top",
				feature: {
					dataZoom: {},
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
			d.push({
				name: entry.protein_ID,
				value: [entry.tsne_1, entry.tsne_2],
				itemStyle: {
					color: CATEGORY_COLORS[SUBCATEGORY_MAP[entry.top1]] || NA_COLOR,
				},
			});
		});
		option.series.push({
			type: "scatter",
			name: "User Data",
			symbolSize: 10,
			itemStyle: {
				color: NA_COLOR,
				borderColor: "black",
				borderWidth: 1,
				opacity: 1,
			},
			data: d,
			zlevel: 12, // Render the "User Data" series above the other series to ensure visibility.
			markLine: {
				symbol: ["none", "none"],
				data: [], // This will be populated dynamically when a user data point is clicked.
				z: 100,
			},
		});

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
		var d = [];
		var isFirst = true;
		for (const nn of nearestNeighbors[clicked.protein_ID]) {
			var nn_metadata = metadataTable.entry(nn.protein_ID);

			d.push([
				{
					coord: [clicked.tsne_1, clicked.tsne_2],
				},
				{
					name: nn.protein_ID,
					coord: [nn_metadata.tsne_1, nn_metadata.tsne_2],
					value: nn.pca_distance,
					label: {
						show: isFirst,
						formatter: (params) => {
							return params.name;
						}
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
						type: isFirst ? "solid" : "dashed",
						width: isFirst ? 1.5 : 1,
					},
				},
			]);

			if (isFirst) isFirst = false;
		}

		// Set mark line data of "User Data" series to connect the clicked point to its nearest neighbor.
		this.echart.setOption({
			series: this.echart.getOption().series.map((series) => {
				if (series.name === "User Data") {
					return {
						...series,
						markLine: {
							data: d,
						},
					};
				}
				return series;
			}),
		});
	}

	resetHighlight() {
		// Clear the mark line data of the "User Data" series to remove any existing highlight.
		this.echart.setOption({
			series: this.echart.getOption().series.map((series) => {
				if (series.name === "User Data") {
					return {
						...series,
						markLine: {
							data: [],
						},
					};
				}
				return series;
			}),
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
			const maxRange = 300; // Hard coded based on initial chart range. Adjust as necessary for new data.
			const xRange = params.batch[0].endValue - params.batch[0].startValue; // Get the current x-axis range.
			const yRange = params.batch[1].endValue - params.batch[1].startValue; // Get the current y-axis range.
			const zoomLevel = (Math.min(xRange, yRange) / maxRange).toFixed(2); // Calculate the zoom level based on the smaller of the two ranges.
			let newSymbolSize;
			if (isNaN(zoomLevel)) {
				newSymbolSize = 1; // Default symbol size if zoom level is not a number.
			} else {
				newSymbolSize = Math.min(4, Math.round(1 / zoomLevel)); // Adjust symbol size inversely proportional to zoom level, capped at a maximum size of 6.
			}
			this.echart.setOption({
				series: this.echart.getOption().series.map((series) => ({
					...series,
					symbolSize: series.name == "User Data" ? 10 : newSymbolSize,
				})),
			});
		});
	}

	/**
	 * Formats the tooltip content for the embedding landscape chart based on the series and data point being hovered over.
	 *
	 * @param {object} params ECharts event parameters for the `tooltip` event.
	 * @returns {string|undefined} The formatted tooltip content as an HTML string, or undefined if the tooltip should not be displayed.
	 */
	#formatTooltip(params) {
		if (params.data === undefined || params.componentType !== "series") {
			return;
		}
		if (params.seriesName === "User Data") {
			if (!userdataTable) return;
			const info = userdataTable.entry(params.data.name);
			const nearestNeighbor = nearestNeighbors[params.data.name][0];
			var content = `<code>ID: ${params.data.name}</code><br>`;
			let top1 = info.top1;
			let top1Prob = (info["P(top1)"] * 100).toFixed(2);
			content += `<table border=1 frame=void rules=rows>`;
			content += `<tr><td>Description:</td><td><p style="max-width: 300px; font-size: small;">${info.description}</p></td></tr>`;
			content += `<tr><td>Predicted Category:</td><td>${top1} › ${SUBCATEGORY_MAP[top1]} (${top1Prob}%)</td></tr>`;
			content += `<tr><td>Nearest Neighbor:</td><td>${nearestNeighbor.protein_ID} d<sub>PCA</sub>=${nearestNeighbor.pca_distance.toFixed(2)}</td></tr>`;
			content += `</table>`;
			content += `<br><code>Click for more details!</code>`;
		} else {
			if (!metadataTable) return;
			const info = metadataTable.entry(params.data.name);
			var content = `<code>ID: ${params.data.name} | Locus Tag: ${info.locus_tag}</code><br>`;
			content += `<table border=1 frame=void rules=rows>`;
			content += `<tr><td>Organism:</td><td>${info.organism}</td></tr>`;
			content += `<tr><td>Category:</td><td>${info.category} › ${info.subcategory}</td></tr>`;
			content += `<tr><td>Product:</td><td>${info.product}</td></tr>`;
			// content += `<tr><td>t-SNE Coordinates:</td><td>${info.tsne_1.toFixed(2)}, ${info.tsne_2.toFixed(2)}</td></tr>`;
			content += `</table>`;
		}
		return content;
	}

	/**
	 * Toggles the visibility of the legend in the embedding landscape chart.
	 *
	 * Currently not in use.
	 */
	#toggleLegend() {
		if (this.echart.getOption().legend[0].show) {
			this.echart.setOption({
				legend: {
					show: false,
				},
			});
		} else {
			this.echart.setOption({
				legend: {
					show: true,
				},
			});
		}
	}
}
