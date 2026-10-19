"""Step 3: filter out low-quality documents.

    python scripts/filter_data.py
 Change the numbers, re-run, and look at how many
documents each rule removed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_filter import DataFiltering

INPUT = Path("data/cleaned/cosmopedia.jsonl")
OUTPUT = Path("data/filtered/cosmopedia.jsonl")

data_filter = DataFiltering(
    min_chars=100,
    max_chars=100000,
    min_alpha_ratio=0.30,
    max_repeated_line_ratio=0.30,
)
data_filter.run(INPUT, OUTPUT)