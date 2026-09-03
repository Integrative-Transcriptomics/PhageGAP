import { structureColors } from "./sequenceAlignment.js";
import { metadataTable } from "../main.js";
import { NA_COLOR, CATEGORY_COLORS } from "../constants.js";

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
			backgroundColor: "#f7f9fc",
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
		this.glviewer.render();
	}

	setInfo(proteinId, plddt_mean = null, ptm = null, compareCategory = null) {
		var c = `<code>${proteinId}</code>`;
		if (compareCategory !== null) {
			var proteinMetadata = metadataTable.entry(proteinId);
			var bgClr = CATEGORY_COLORS[proteinMetadata.category] || NA_COLOR;
			c += `<code class="reduce-3 ml-1 va-middle" style="background: ${bgClr}">${proteinMetadata.subcategory}</code>`;
		};
		if (plddt_mean !== null) c += `<br/>Mean pLDDT: ${plddt_mean.toFixed(2)}`;
		if (ptm !== null) c += `<br/>Predicted TM-score: ${ptm.toFixed(2)}`;
		document.getElementById("structure-view-info").innerHTML = c;
	}
}
