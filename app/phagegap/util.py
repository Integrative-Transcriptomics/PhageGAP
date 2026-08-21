"""Utility functions for the PhageGap web application."""

from __future__ import annotations

from io import StringIO
from Bio.SeqUtils import seq1
from Bio import Align
from Bio.Align import substitution_matrices


class GFFParser:
	""" A simple GFF parser that reads a GFF file from a StringIO object and returns a list of simplified features. 
	"""
	def __init__(self, gff_string_io: StringIO):
		""" Initializes the GFFParser with a StringIO object containing the GFF file content.
		
		Parameters
		__________
		gff_string_io : StringIO
			A StringIO object containing the GFF file content.
		"""
		self.gff_io = gff_string_io

	def parse(self) -> list[dict]:
		""" Parses the GFF file and returns a list of features.

		Returns
		_______
		list of dict
			A list of dictionaries, each representing a feature in the GFF file.
		"""
		features = []
		with self.gff_io as f:
			for line in f:
				# Only process non-comment lines and lines.
				if line.startswith('#'):
					continue

				# Extract the relevant fields from the GFF line.
				parts = line.strip().split('\t')

				# Only process lines that have the expected number of columns for a GFF file.
				if len(parts) != 9:
					continue

				# Only process lines that represent coding sequences (CDS).
				if parts[2] != "CDS":
					continue
				
				feature = {
					'seqid': parts[0],
					'source': parts[1],
					'type': parts[2],
					'start': int(parts[3]),
					'end': int(parts[4]),
					'score': parts[5],
					'strand': parts[6],
					'phase': parts[7],
					'attributes': self._parse_attributes(parts[8])
				}
				features.append(feature)
		return features

	def _parse_attributes(self, attribute_string: str) -> dict:
		""" Parses the attributes column of a GFF file and returns a dictionary of attributes.

		Parameters
		__________
		attribute_string : str
			A string containing the attributes column of a GFF file.

		Returns
		_______
		dict
			A dictionary of attributes parsed from the attribute string.
		"""
		attributes = {}
		for attribute in attribute_string.split(';'):
			key_value = attribute.split('=')
			if len(key_value) == 2:
				key, value = key_value
				attributes[key] = value
		return attributes


def read_sequence_from_cif(cif_text: str) -> str:
	"""Reads the amino acid sequence from the content of a CIF file.

	Parameters
	__________
	cif_text : str
		The content of a CIF file as a string.

	Returns
	_______
	str
		The amino acid sequence in one-letter code.
	"""
	seq_dict = {}
	atom_site_index = {}
	i = 0
	for line in cif_text.splitlines():
		if line.startswith("_atom_site."):
			label = line.strip()
			atom_site_index[label] = i
			i += 1
		if line.startswith("ATOM"):
			fields = line.split()
			position = int(fields[atom_site_index["_atom_site.label_seq_id"]])
			aa = fields[atom_site_index["_atom_site.label_comp_id"]]
			seq_dict.setdefault(position, aa)
	seq3_str = ''.join( seq_dict.values() )
	seq1_str = seq1(seq3_str)
	return seq1_str


def align_sequences(seq1: str, seq2: str) -> tuple[str, str]:
	"""Aligns two amino acid sequences using BLOSUM substitution matrices and returns the best alignment.

	Iteratively aligns the sequences using BLOSUM80, BLOSUM62, and BLOSUM50 substitution matrices and returns
	the alignment with the highest score.

	Parameters
	__________
	seq1 : str
		The first amino acid sequence in one-letter code.
	seq2 : str
		The second amino acid sequence in one-letter code.

	Returns
	_______
	tuple
		A tuple containing the aligned sequences with the highest alignment score.
	"""
	aligner = Align.PairwiseAligner()
	aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
	alignment = aligner.align(seq1, seq2)
	identity = _compute_identity(alignment[0][0], alignment[0][1])

	if identity >= 0.75:
		aligner.substitution_matrix = substitution_matrices.load("BLOSUM90")
		alignment = aligner.align(seq1, seq2)
	elif identity < 0.30:
		aligner.substitution_matrix = substitution_matrices.load("BLOSUM50")
		alignment = aligner.align(seq1, seq2)
	return alignment[0][0], alignment[0][1]


def _compute_identity(seq1: str, seq2: str) -> float:
	"""Computes the identity between two aligned sequences.

	Parameters
	__________
	seq1 : str
		The first aligned amino acid sequence in one-letter code.
	seq2 : str
		The second aligned amino acid sequence in one-letter code.

	Returns
	_______
	float
		The identity between the two sequences as a fraction of identical residues over the total length of the alignment.
	"""
	if len(seq1) != len(seq2):
		raise ValueError("Aligned sequences must have the same length.")
	matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
	return matches / len(seq1)
