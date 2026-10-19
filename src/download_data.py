""" Step 1, need to download data
    data I have found is related to climate and 
    this datset consists text
    in this model we will you only text sections
    as this model is next token prediction, we will do chatting in the next model

"""

import sys
from pathlib import Path
from datasets import load_dataset

# Let Python find the src/ folder when this script is run from the repo root.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.jsonl_io import write_jsonl

class DataDownloader:
    def __init__(self, split = str('train')):
        # self.dataset_name = dataset_name
        self.split = split

    def run(self, output_path, max_documents=None):
        """Download the dataset and save it as JSONL.

        max_documents=70000 takes only the first 50k stories. The full set is
        over 2 million - fine for a real run, far too slow for a lesson where
        we want to see the whole pipeline work end to end in a few minutes.
        """
        output_path = Path(output_path)

        # print("Downloading:", "HuggingFaceTB/cosmopedia"| "split:", self.split)

        # load_dataset caches on disk, so running this a second time
        # downloads nothing.
        dataset = load_dataset("HuggingFaceTB/cosmopedia", "auto_math_text", split=self.split) 

        documents = []
        for i, item in enumerate(dataset):
            if max_documents is not None and i >= max_documents:
                break
            # We keep only the text. Anything else the dataset carries is not
            # used by a pretraining pipeline.
            documents.append({"text": item["text"]})

        write_jsonl(output_path, documents)

        print("Saved", len(documents), "documents to", output_path)
        return len(documents)