import { Chart } from "./chart.js";
import { userdataTable, selectedId } from "../main.js";
import { displayNotification } from "../utility.js";
import { CATEGORY_COLORS, SUBCATEGORY_MAP, NA_COLOR } from "../constants.js";

export class GenomicContext extends Chart {
	/**
	 * Constructor for the GenomicContext class, which extends the base Chart class.
	 *
	 * @param {element} element The DOM element to bind the chart to.
	 */
	constructor(element) {
		super(element);
		this.features = [];
		this.featuresIndex = {};
	}

	fill() {
		if (!this.echart) return;
		if (!featureTable) return;

		// Clear features and featuresIndex before populating them with new data.
		this.features = [];
		this.featuresIndex = {};
		let idx = 0;

		// Populate the features array and featuresIndex mapping from the featureTable.
		featureTable.entries().forEach((item) => {
			let feature = {};
			let category = "Unknown"; // Default category for features without synchronized user data.
			let clr = NA_COLOR; // Default color for features without synchronized user data.

			// Populate feature properties from the feature table entry.
			feature["name"] = item.attributes["Name"];
			feature["locus_tag"] = item.attributes["locus_tag"];
			feature["strand"] = item.strand;
			feature["product"] = item.attributes["product"];
			feature["value"] = [
				feature["strand"] === "+" ? 0 : 1,
				item.start,
				item.end,
				feature["name"],
			];
			this.featuresIndex[feature["name"]] = idx;
			idx++;

			// If userdataTable is available, attempt to retrieve user data for the feature and update its category and color accordingly.
			if (userdataTable) {
				var entry = userdataTable.entry(feature["name"]);
				if (entry) {
					category = entry.top1 || category;
					clr = CATEGORY_COLORS[SUBCATEGORY_MAP[category]] || clr;
				}
			}

			// Set the feature's category and itemStyle properties based on the determined category and color.
			feature["category"] = category;
			feature["itemStyle"] = {
				color: clr,
				opacity: 1,
				borderColor: "#ffffff",
				borderWidth: 1,
			};

			// Add the feature to the features array for rendering in the chart.
			this.features.push(feature);
		});

		const option = {
			grid: {
				containLabel: true,
				left: "4%",
				top: "5%",
				right: "4%",
				bottom: "15%",
			},
			xAxis: {
				name: "Position",
				nameLocation: "center",
				nameGap: 20,
				min: 1,
			},
			yAxis: {
				name: "Strand",
				nameLocation: "center",
				nameGap: 35,
				data: [0, 1],
				axisLabel: {
					fontSize: 18,
					formatter: (value) => (value == 0 ? "+" : "-"),
				},
				inverse: true,
				min: 0,
				max: 1,
			},
			dataZoom: [
				{
					type: "inside",
					xAxisIndex: 0,
				},
			],
			tooltip: {
				trigger: "item",
				formatter: this.#formatTooltip,
			},
			series: [
				{
					type: "custom",
					renderItem: this.#render,
					encode: {
						x: [1, 2],
						y: 0,
					},
					data: this.features,
				},
			],
		};

		// Set the prepared option object to the ECharts instance and activate the chart.
		this.state.setValue(null);
		this.echart.setOption(option);
		this.zoom();

		// Display user notification about successful loading of the genomic context chart.
		displayNotification(
			`Loaded ${this.features.length} features into <code>Genomic Context</code>.`,
			"Success",
			"success",
		);
	}

	zoom() {
		if (!this.echart) return;
		if (!featureTable) return;
		let selectedProteinID = selectedId.getValue();

		if (selectedProteinID == null) {
			// Reset zoom, if selected is null.
			this.echart.dispatchAction({
				type: "dataZoom",
				start: 0,
				end: 100,
			});
			return;
		} else {
			// Zoom to the selected feature if it exists in the featuresIndex mapping.
			if (!this.featuresIndex.hasOwnProperty(selectedProteinID)) return;

			const idx = this.featuresIndex[selectedProteinID];
			const leftIdx = Math.max(0, idx - 5);
			const rightIdx = Math.min(this.features.length - 1, idx + 5);
			this.echart.dispatchAction({
				type: "dataZoom",
				startValue: this.features[leftIdx]["value"][1],
				endValue: this.features[rightIdx]["value"][2],
			});
		}
	}

	#formatTooltip(params) {
		let d = params.data;
		return `
			<div style="font-size: 14px; font-weight: bold; margin-bottom: 5px;">
				<code>ID: ${d.name} | Locus Tag: ${d.locus_tag}</code>
			</div>
			<strong>Start:</strong> ${d.value[1]}<br/>
			<strong>End:</strong> ${d.value[2]}<br/>
			<strong>Strand:</strong> ${d.strand}<br/>
			<strong>Product:</strong> ${d.product}<br/>
			<strong>Predicted Category:</strong> ${d.category}
		`;
	}

	#render(params, api) {
		// Extract the necessary data for rendering the feature from the API.
		const strand = api.value(0);
		const start = api.coord([api.value(1), strand]);
		const end = api.coord([api.value(2), strand]);
		const height = api.size([0, 1])[1] * 0.6;
		const width = end[0] - start[0];
		const label = api.value(3);
		const isSelected = selectedId.getValue() === label; // Check if the current feature is the selected one.

		// Construct the shape of the feature as a rectangle and clip it to the chart's coordinate system to ensure it doesn't overflow.
		const shape = echarts.graphic.clipRectByRect(
			{
				x: start[0],
				y: start[1] - height / 2,
				width: width,
				height: height,
				r: 2, // Rounded corners for the feature rectangle.
			},
			{
				x: params.coordSys.x,
				y: params.coordSys.y,
				width: params.coordSys.width,
				height: params.coordSys.height,
				r: 2, // Rounded corners for the clipping rectangle.
			},
		);

		// Construct list of elements to render.
		const children = [
			{
				type: "rect",
				shape: shape,
				style: {
					fill: api.visual("color"),
					stroke: isSelected ? "#333333" : "#f7f9fc", // Highlight the border if the feature is selected.
					lineWidth: isSelected ? 2 : 1, // Thicker border for selected feature.
					strokeNoScale: true,
				},
			},
		];

		// Check if the label can fit within the feature's width before adding it to the children array for rendering.
		const font = "11px sans-serif";
		const textRect = echarts.format.getTextRect(label, font);
		if (textRect.width + 10 <= width) {
			children.push({
				type: "text",
				style: {
					text: label,
					x: start[0] + width / 2,
					y: start[1],
					textAlign: "center",
					textVerticalAlign: "middle",
					fontFamily: "monospace",
					fontSize: 11,
					fill: "#111",
				},
			});
		}

		// Return a group containing the feature rectangle and, if applicable, the label text.
		return {
			type: "group",
			children: children,
		};
	}
}

/*
The following methods are not chart related, but to store the features table and to initialize it.
This is done here because the features table is only used in the genomic context chart.
*/

import { ArqueroTable } from "../table.js";

/**
 * The features table is an Arquero table that stores the features data for the genomic context chart.
 */
export var featureTable = null;

/**
 * Initializes the features table with the provided data. This function should be called after loading a GFF
 * file to populate the features table.
 * 
 * @param {any} data See https://idl.uw.edu/arquero/api/#from for the expected data format.
 * @returns {Promise} A promise that resolves when the features table is initialized.
 */
export function populateFeatures(data) {
	// Return promise that resolves when the table is initialized.
	return new Promise((resolve, reject) => {
		try {
			// Fill the table with the provided data using Arquero.
			featureTable = new ArqueroTable(data);
			resolve();
		} catch (error) {
			console.error(error); // This should not fail.
			reject(error);
		}
	});
}
