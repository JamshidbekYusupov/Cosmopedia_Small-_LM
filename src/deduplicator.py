"""Stage 4 - deduplication."""

import hashlib
from pathlib import Path

from src.jsonl_io import read_jsonl, write_jsonl

class Deduplicator:
    def shingles(self, text, k = 5):
        words = text.split()
        return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}

    def minhash_signature(self, shingle_set, num_hashes = 8):
        # A short fingerprint that approximates shingle overlap.
        if not shingle_set:
            return tuple([0] * num_hashes)
        return tuple(
            min(hash((salt, s))for s in shingle_set)
            for salt in range(num_hashes)
        )

    def near_duplication(self, documents):

        signatures = [
            self.minhash_signature(self.shingles(doc['text'])) 
            for doc in documents
        ]

        seen = set()
        near_dupes = 0 
        for signature in signatures:
            if signature in seen:
                near_dupes  += 1
            seen.add(signature)
        return near_dupes

    
    def document_hash(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def run(self, input_path, output_path, limit = None):

        documents = read_jsonl(Path(input_path), limit=limit)

        seen  = set()
        unique = []
        duplicates = 0

        for doc in documents:
            fingerprint = self.document_hash(doc['text'])

            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            unique.append(doc) 
        write_jsonl(Path(output_path), unique)

        near_dupes = self.near_duplication(unique)
        stats = {
            "input": len(documents),
            "duplicates_removed": duplicates,
            "near_duplicates_found": near_dupes,
            "output": len(unique),
        }
        print("Deduplication:", stats)
        return stats