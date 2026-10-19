import json 
import random 
import time

from pathlib import Path

import numpy as np 

from src.jsonl_io import read_jsonl, write_jsonl


DTYPE = np.uint16


class DatasetBuilder:
    def __init__(self, tokenizer, val_ratio = 0.01, seed = 1337):
        self.tokenizer = tokenizer
        self.val_ratio = val_ratio
        self.seed = seed

    def split_documents(self, documents):
        indices = list(range(len(documents)))

        random.Random(self.seed).shuffle(indices)

        n_val = max(1, int(len(documents) * self.val_ratio))

        val_docs = [documents[i] for i in indices[: n_val]]
        train_docs = [documents[i] for i in indices[n_val:]]

        return train_docs, val_docs

    def encode_to_bin(self, documenst, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        eos_id = self.tokenizer.eos_id()
        n_tokens = 0

        with open(output_path, 'wb') as f:
            for doc in documenst:
                ids = self.tokenizer.encode(doc["text"])
                ids.append(eos_id)

                array = np.array(ids, dtype=DTYPE)

                array.tofile(f)
                n_tokens += array.size
        
        print(output_path.name, "->", n_tokens, "tokens")
        return n_tokens

    def get_batch(self, bin_path, batch_size, block_size):

        data = np.memmap(bin_path, dtype=DTYPE, mode = "r")
        starts = np.random.randint(0, len(data) - block_size -  1, size=batch_size)

        x = np.stack([data[i:i+block_size].astype(np.int64) for i in starts])

        y = np.stack([data[i+1: i + 1 + block_size].astype(np.int64) for i in starts])

        return x, y

    def run(self, input_path, train_bin, val_bin, 
            meta_path, block_size = 256, limit = None):
        documents = read_jsonl(Path(input_path), limit=limit)
        print("Documenst:", len(documents))

        train_docs, val_docs = self.split_documents(documents)
        # print("Split ->", len(train_docs), "train /", len(val_docs), "val")

        n_train = self.encode_to_bin(train_docs, train_bin)
        n_val = self.encode_to_bin(val_docs, val_bin)

        meta = {
            "vocab_size": self.tokenizer.vocab_size(),
            "eos_id": self.tokenizer.eos_id(),
            "dtype": np.dtype(DTYPE).name,
            "block_size": block_size,
            "seed": self.seed,
            "val_ratio": self.val_ratio,
            "n_docs_train": len(train_docs),
            "n_docs_val": len(val_docs),
            "n_tokens_train": int(n_train),
            "n_tokens_val": int(n_val),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        meta_path = Path(meta_path)

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            print("Saved", meta_path)

            return meta
        