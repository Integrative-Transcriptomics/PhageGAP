"""
Utility functions for the package.
"""

from __future__ import annotations

import random, torch, logging
import numpy as np
from transformers import set_seed

logger = logging.getLogger(__name__)


def set_determinism(seed: int) -> None:
	"""Set seeds for python, numpy, transformers and torch.

	Parameters
	----------
	seed (int):
		The seed value to set for all random number generators.

	Raises
	-------
	AssertionError
		If seed is negative.
	"""
	assert seed >= 0, "Seed must be non-negative."

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	set_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True, warn_only=True)
