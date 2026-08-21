import { Observable } from "../observable.js";

/**
 * Class to build and manage ECharts instances for various chart types in the application.
 */
export class Chart {
	/**
	 * Constructor for the Chart class.
	 *
	 * @param {element} element The DOM element to bind the chart to.
	 */
	constructor(element) {
		this.echart = echarts.init(element, {
			width: "auto",
			height: "auto",
		});
		this.observer = new ResizeObserver(() => {
			requestAnimationFrame(() => {
				this.resize();
			});
		}).observe(element);
		this.state = new Observable(null);
		this.state.onChange(() => {
			let value = this.state.getValue();
			console.log("Chart state changed: ", value);
			if (value == null) {
				this.setActive();
			} else {
				this.setLoading(value);
			}
		});
	}

	/**
	 * Sets the chart to a loading state with a custom message.
	 *
	 * @param {string} message The message to display while the chart is loading data.
	 */
	setLoading(message) {
		this.echart.showLoading({
			text: message,
			textColor: "#666666",
			color: "transparent",
			maskColor: "transparent",
			showSpinner: false,
			zlevel: 0,
		});
	}

	/**
	 * Sets the chart to an active state, hiding any loading indicators.
	 */
	setActive() {
		this.echart.hideLoading();
	}

	/**
	 * Resizes the chart to fit its container. Should be called when the window is resized or when the container size changes.
	 */
	resize() {
		this.echart.resize({
			width: "auto",
			height: "auto",
		});
	}
}
