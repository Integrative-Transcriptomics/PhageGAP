import { Chart } from "./chart.js";
import { NA_COLOR } from "../constants.js";

const ALIGNMENT_COLORS = {
	M: "#364B9A",
	X: "#A01813",
	D: "#DDDDDD",
	I: "#BBBBBB",
	NA: "#F7F9FC",
};

export var structureColors = [];

export class SequenceAlignment extends Chart {
	constructor(element) {
		super(element);
	}

	fill(seqA, seqB, labelA = "Sequence 1", labelB = "Sequence 2") {
		if (!this.echart) return;
		if (!seqA || !seqB || seqA.length !== seqB.length) return;

		const option = {
			grid: {
				left: "8%",
				right: "1%",
				top: 10,
				bottom: "6%",
			},
			tooltip: {
				trigger: "axis",
				axisPointer: {
					axis: "x",
				},
				formatter: (params) => {
					const p1 = params[0].data;
					const p2 = params[1].data;
					return `
					<table border=1 frame=void rules=rows>
						<tr>
							<td>${p1.sequenceName}</td>
							<td>${p1.position !== null ? p1.position : ""}</td>
							<td>${p1.value[2]}</td>
						</tr>
						<tr>
							<td>${p2.sequenceName}</td>
							<td>${p2.position !== null ? p2.position : ""}</td>
							<td>${p2.value[2]}</td>
						</tr>
					</table>
					`;
				},
			},
			dataZoom: [
				{
					type: "slider",
					xAxisIndex: 0,
					height: 5,
					top: "top",
					filterMode: "weakFilter",
					labelPrecision: 0,
					showDetail: false,
					moveHandleSize: 1,
					handleSize: "300%",
					handleStyle: {
						color: "#444444",
					},
					backgroundColor: "transparent",
					fillerColor: "rgba(6, 36, 101, 0.1)",
				},
			],
			xAxis: {
				type: "value",
				min: 0.5,
				max: seqA.length + 0.5,
				minInterval: 1,
				splitLine: { show: false },
				axisLabel: {
					formatter: (value) => Number(value),
					fontSize: 11,
					showMinLabel: false,
					showMaxLabel: false,
				},
			},
			yAxis: {
				type: "category",
				data: [labelB, labelA],
				axisTick: { show: false },
				axisLabel: {
					fontSize: 11,
				},
			},
			series: [
				{
					type: "custom",
					coordinateSystem: "cartesian2d",
					encode: {
						x: 0,
						y: 1,
						c: 2,
					},
					data: this.#toAlignmentData(seqA, seqB, labelA, labelB),
					renderItem: this.#render,
				},
			],
		};

		this.echart.setOption(option);
	}

	clear() {
		if (!this.echart) return;
		this.echart.clear();
	}

	setIdentity(identity) {
		var label = document.getElementById("sequence-alignment-identity-label");
		// Set the sequence identity information for the sequence alignment chart.
		if (identity === null || identity === undefined) {
			label.innerHTML = "";
		} else {
			if (identity > 40) {
				label.style.backgroundColor = "#BBDDBB"; // Pale green.
			} else if (identity < 20) {
				label.style.backgroundColor = "#FFBBCC"; // Pale red.
			} else {
				label.style.backgroundColor = "#EEEEBB"; // Pale yellow.
			}
			label.innerHTML = `Identity: ${identity}%`;
		}
	}

	#state(charA, charB) {
		if (charA === charB) {
			return ["M", "M"];
		} else {
			if (charA === "-") {
				// Insertion
				return ["NA", "I"];
			} else if (charB === "-") {
				// Deletion
				return ["D", "NA"];
			} else {
				return ["X", "X"];
			}
		}
	}

	#toAlignmentData(seqA, seqB, labelA = "Sequence 1", labelB = "Sequence 2") {
		const data = [];
		structureColors = []; // Empty the structureColors array.
		var posA = 1; // The sequence position for Sequence A, starting from 1.
		var posB = 1; // The sequence position for Sequence B, starting from 1.
		for (let i = 0; i < seqA.length; i++) {
			const charA = seqA[i];
			const charB = seqB[i];
			var state = this.#state(charA, charB);

			let colorA = ALIGNMENT_COLORS[state[0]];
			if (charA !== "-") {
				structureColors.push(colorA); // Store the color for Sequence A in the structureColors array.
			}
			data.push({
				value: [i + 1, 1, charA],
				position: charA === "-" ? null : posA,
				row: 1,
				sequenceName: labelA,
				state: state[0],
				itemStyle: {
					color: colorA,
				},
			});
			if (charA !== "-") posA++; // Increment position for Sequence A if it's not a gap.

			let colorB = ALIGNMENT_COLORS[state[1]];
			data.push({
				value: [i + 1, 0, charB],
				position: charB === "-" ? null : posB,
				row: 0,
				sequenceName: labelB,
				state: state[1],
				itemStyle: {
					color: colorB,
				},
			});
			if (charB !== "-") posB++; // Increment position for Sequence B if it's not a gap.
		}
		return data;
	}

	#render(params, api) {
		const xValue = api.value(0);
		const yValue = api.value(1);
		const text = String(api.value(2));

		const center = api.coord([xValue, yValue]);
		const size = api.size([1, 1]);

		const width = size[0] * 0.95;
		const height = size[1] * 0.8;

		const rectShape = echarts.graphic.clipRectByRect(
			{
				x: center[0] - width / 2,
				y: center[1] - height / 2,
				width,
				height,
			},
			{
				x: params.coordSys.x,
				y: params.coordSys.y,
				width: params.coordSys.width,
				height: params.coordSys.height,
			},
		);
		const children = [
			{
				type: "rect",
				shape: rectShape,
				style: api.style(),
			},
		];

		// Check if the label can fit within the cell's width before adding it to the children array for rendering.
		const font = `11px monospace`;
		const textRect = echarts.format.getTextRect(text, font);
		if (textRect.width + 4 <= width) {
			children.push({
				type: "text",
				style: {
					text,
					x: center[0],
					y: center[1],
					textAlign: "center",
					textVerticalAlign: "middle",
					fontFamily: "monospace",
					fontSize: 11,
					fill: "#111",
				},
			});
		}

		return {
			type: "group",
			children,
		};
	}
}
