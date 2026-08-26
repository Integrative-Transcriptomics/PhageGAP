import * as util from "../utility.js";
import { Chart } from "./chart.js";
import { CATEGORY_COLORS, SUBCATEGORY_MAP, NA_COLOR } from "../constants.js";

/**
 * Class to build and manage the category probabilities chart using ECharts.
 *
 * Extends the base {@link Chart} class to provide specific functionality for displaying category probabilities
 * for a selected protein in the embedding landscape chart. The chart visualizes the top 3 predicted categories
 * and their associated probabilities as a horizontal bar chart.
 */
export class CategoryProbabilities extends Chart {
	/**
	 * Constructor for the CategoryProbabilities class, which extends the base Chart class.
	 *
	 * @param {element} element The DOM element to bind the chart to.
	 */
	constructor(element) {
		super(element);
	}

	/**
	 * Shows the category probabilities for a specific protein in the embedding landscape chart.
	 *
	 * @param {object} clicked An entry from the `userdataTable` corresponding to the clicked data point in the embedding landscape chart.
	 */
	show(clicked) {
		if (!this.echart) return;

		// Extract the top 3 categories and their probabilities from the entry.
		let d = [
			[clicked.top1, clicked["P(top1)"]],
			[clicked.top2, clicked["P(top2)"]],
			[clicked.top3, clicked["P(top3)"]],
		];

		// Refactor the data to prepare it for visualization in the bar chart.
		let categories = [];
		let probabilities = [];
		d.toSorted((a, b) => b[1] - a[1]) // Ensure the categories are sorted by probability in descending order.
			.forEach((item) => {
				categories.push(util.titleize(item[0])); // Convert category name to title case.
				probabilities.push({
					value: (item[1] * 100).toFixed(2), // Convert to percentage and round to 2 decimal places.
					itemStyle: {
						color: CATEGORY_COLORS[SUBCATEGORY_MAP[item[0]]] || NA_COLOR, // Use predefined color or default to gray.
					},
				});
			});

		const option = {
			grid: {
				containLabel: true,
				left: "5%",
				top: "5%",
				right: "1%",
				bottom: "1%",
			},
			xAxis: {
				show: false,
				min: 0,
				max: 100,
			},
			yAxis: {
				name: "",
				nameLocation: "center",
				data: categories,
				inverse: true,
			},
			series: {
				type: "bar",
				name: "Category Probabilities",
				label: {
					show: true,
					formatter: "{c}%",
					offset: [20, 0],
					distance: 0,
				},
				showBackground: true,
				data: probabilities,
			},
		};

		// Set the chart option to render the bar chart with the prepared data and configuration.
		this.echart.setOption(option);
		this.setActive();
	}

	clear() {
		if (!this.echart) return;
		this.echart.clear();
	}
}
