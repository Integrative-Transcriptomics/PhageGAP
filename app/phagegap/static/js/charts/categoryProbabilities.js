import * as util from "../utility.js";
import { Chart } from "./chart.js";
import { userdataTable, nearestNeighbors } from "../main.js";
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
	showDetail(clicked) {
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

	/**
	 * Displays a summary of function prediction probabilities of all proteins
	 * in a stacked bar chart.
	 */
	showSummary() {
		if (!this.echart) return;

		const probabilityCountsNovel = {};
		const probabilityCountsKnown = {};

		for (let i = 1; i <= 100; i++) {
			probabilityCountsNovel[i.toString()] = 0;
			probabilityCountsKnown[i.toString()] = 0;
		}

		for (const entry of userdataTable.entries()) {
			const p = (entry["P(top1)"] * 100).toFixed(0);

			if (nearestNeighbors[entry.protein_ID][0].pca_distance < 0.001) {
				probabilityCountsKnown[p]++;
			} else {
				probabilityCountsNovel[p]++;
			}
		}

		// Style definitions for background elements like legend, toolbox, and tooltip to enhance visibility.
		const backgroundStyle = {
			backgroundColor: "rgba(247, 249, 252, 0.9)", // Light blue background for better visibility.
			borderRadius: 5,
		};

		const option = {
			grid: {
				containLabel: true,
				left: "5%",
				top: "10%",
				right: "2%",
				bottom: "10%",
			},
			xAxis: {
				type: "category",
				data: Object.keys(probabilityCountsNovel),
				interval: 10,
				name: "Probability (%)",
				nameGap: 20,
				nameLocation: "center",
			},
			yAxis: {
				type: "value",
				name: "Count",
				nameGap: 25,
				nameLocation: "center",
			},
			legend: {
				orient: "horizontal",
				top: 0,
				left: "center",
				textStyle: {
					fontSize: 10,
				},
				itemWidth: 5,
				itemHeight: 15,
				selectedMode: false, // Disable toggling of series visibility by clicking on legend items.
				show: true,
				...backgroundStyle,
			},
			series: [
				{
					type: "bar",
					name: "Non-identical",
					data: Object.values(probabilityCountsNovel),
					stack: "total",
					itemStyle: {
						color: "#062465",
					},
				},
				{
					type: "bar",
					name: "Reference-identical",
					data: Object.values(probabilityCountsKnown),
					stack: "total",
					itemStyle: {
						color: "#888888",
					},
				},
			],
		};

		this.echart.setOption(option);

		this.setActive();
	}

	/**
	 * Clears the category probabilities chart, removing any displayed data and resetting the chart state.
	 */
	clear() {
		if (!this.echart) return;
		this.echart.clear();
	}
}
