"""Step 1: download the raw corpus.

    python scripts/download_data.py
"""

import sys
from pathlib import Path

# Let Python find the src/ folder when this script is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.download_data import DataDownloader

# Change this to download more or fewer stories. None = the whole dataset
# (over 2 million documents - slow, only do it when you mean it).
MAX_DOCUMENTS = 70000

# from datasets import load_dataset

# ds = load_dataset("HuggingFaceTB/cosmopedia", "auto_math_text")

OUTPUT = Path("data/raw/cosmopedia.jsonl")

downloader = DataDownloader(split='train')
downloader.run(OUTPUT, max_documents=MAX_DOCUMENTS)