"""
This is 2nd step after downloading datset, we will start cleaning it

"""

import re, unicodedata
from pathlib import Path

from src.jsonl_io import write_jsonl, read_jsonl

class DataCleaner:
    def clean_text(self, text):
        text = self.remove_html(text)
        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)
        text = self.normalize_whitespace(text)

        return text
    
    def remove_html(self, text):
        return re.sub(r"<[^>]+>", " ", text)
    
    def normalize_unicode(self, text):
        return unicodedata.normalize("NFKC", text)
    
    def remove_control_characters(self, text):
        return "".join(
            char for char in text
            if char in "\n\t" or not unicodedata.category(char).startswith("C")
        )
    
    def normalize_whitespace(self, text):
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()
    
    def run(self, input_path, output_path, limit = None):
        documents = read_jsonl(Path(input_path), limit=limit)
        cleaned = []
        empty_removed = 0
        for doc in documents:
            text = self.clean_text(doc['text'])
            if not text:
                empty_removed += 1
                continue
            cleaned.append({"text":text})
        write_jsonl(Path(output_path), cleaned)

        stats = {
            "input": len(documents),
            "empty_removed": empty_removed,
            "output": len(cleaned)

        }
        print("Cleaning:", stats)
        return stats
    