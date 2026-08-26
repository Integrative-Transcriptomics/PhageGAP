import { structureColors } from "./sequenceAlignment.js";
import { NA_COLOR } from "../constants.js";

/**
 * Internal class for handling the structure view.
 */
export class StructureView {
	/**
	 * Class constructor.
	 *
	 * @param {String} domid DOM id of the element to bind the component to.
	 */
	constructor(element) {
		this.glviewer = $3Dmol.createViewer($("#" + element.id), {
			opacity: 0,
			antialias: true,
			cartoonQuality: 6,
		});
	}

	/**
	 * Clears the structure view of all content.
	 */
	clear() {
		this.glviewer.clear();
		this.glviewer.render();
		document.getElementById("structure-view-info").innerHTML = "";
	}

	/**
	 * Adds a protein structure to this view.
	 *
	 * @param {String} data Protein structure in one of the formats supported by `3Dmol.js`.
	 * @param {String} format The format of the provided structure data.
	 */
	fill(data, format) {
		this.glviewer.addModel(data, format);
		this.glviewer.zoomTo();
		var setColor = function (atom) {
			if (!structureColors || structureColors.length == 0) return NA_COLOR;
			return structureColors[atom.resi - 1] || NA_COLOR;
		};
		this.glviewer.setStyle(
			{},
			{
				cartoon: {
					colorfunc: setColor,
				},
			},
		);
		/*this.glviewer.addSurface($3Dmol.SurfaceType.MS, {
			smoothness: 1,
			opacity: 0.5,
			color: NA_COLOR,
		});*/
		this.glviewer.render();
	}

	setInfo(text, plddt_mean = null, ptm = null) {
		var c = `<code>${text}</code>`;
		if (plddt_mean !== null) {
			c += `<br/>Mean pLDDT: ${plddt_mean.toFixed(2)}`;
		}
		if (ptm !== null) {
			c += `<br/>Predicted TM-score: ${ptm.toFixed(2)}`;
		}
		document.getElementById("structure-view-info").innerHTML = c;
	}
}
