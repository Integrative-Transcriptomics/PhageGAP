/**
 * Stores data in an Arquero table and provides methods to access it.
 */
export class ArqueroTable {
	/**
	 * Constructor for the MetadataTable class.
	 * 
	 * @param {any} data See https://idl.uw.edu/arquero/api/#from for the expected data format.
	 */
	constructor(data) {
		this.table = aq.from(data);
	}

	/**
	 * Retrieves a single entry by protein ID.
	 * 
	 * @param {string} id The protein ID to look up in the metadata table.
	 * @returns {object|null} The metadata object for the given protein ID, or null if not found. 
	 */
	entry(id) {
		const matches = this.table
			.params({ id })
			.filter((row, $) => row.protein_ID === $.id);
		return matches.numRows() > 0 ? matches.object() : null;
	}

	/**
	 * Retrieves all entries in the metadata table.
	 * 
	 * @returns {Array} An array of all objects in the table oriented as records.
	 */
	entries() {
		return this.table.objects();
	}

	/**
	 * Retrieves all entries grouped by a specified column.
	 * 
	 * @param {string} column The column name to group the metadata entries by (e.g., 'category').
	 * @returns 
	 */
	groups(column) {
		return this.table
				.groupby(column)
				.objects({ grouped: true });
	}

	insert(data) {
		this.table = this.table.concat(
			aq.from(data).antijoin(this.table, "protein_ID"),
		);
	}
}
