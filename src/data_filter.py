"""
Step 3~ deciding which documents we should keep.
Every rule below is a HEURISTIC,
not a truth: each one will sometimes throw away a document we wanted. That is
the trade, and it is why the run() method counts each reason separately - when
a threshold is wrong, the counts are how you find out.
"""

import re 
from collections import Counter
from pathlib import Path

from src.jsonl_io import write_jsonl, read_jsonl

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

class DataFiltering: 
    def __init__(self, min_chars = 20, max_chars = 1000, min_alpha_ratio = 0.3,
                 max_repeated_line_ratio = 0.3):

        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_alpha_ratio = min_alpha_ratio
        self.max_repeated_line_ratio = max_repeated_line_ratio

    def mask_pii(self, text):

        text, n_email = EMAIL_RE.subn("<EMAIL>", text)
        text, n_phone = PHONE_RE.subn("<Phone Number>", text)

        return text, n_email + n_phone
    
    def alphabetic_ratio(self, text):
        if not text:
            return 0.0
        return sum(char.isalpha() for char in text) / len(text)
    
    def repeated_line_ratio(self, text):
        """Share of lines that appear more than once.

        High values mean boilerplate: a cookie banner, a menu, a template.
        The model would memorise the repetition instead of learning language.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) <= 1:
            return 0.0

        counts = Counter(lines)
        repeated = sum(count for count in counts.values() if count > 1)
        return repeated / len(lines)

    def reject_reason(self, text):
        if len(text) < self.min_chars:
            return "too short"

        if len(text) > self.max_chars:
            return "too long"

        if self.alphabetic_ratio(text) < self.min_alpha_ratio:
            return "low_alpha_ratio"
       
        if self.repeated_line_ratio(text) > self.max_repeated_line_ratio:
            return "repeated_lines"
        return None
    def run(self, input_path, output_path, limit = None):
        documents = read_jsonl(Path(input_path), limit=limit)

        kept = []

        dropped = Counter()

        pii_masked = 0

        for doc in documents:
            text, pii = self.mask_pii(doc['text'])
            pii_masked += pii

            reason = self.reject_reason(text)
            if reason:
                dropped[reason] += 1
                continue
            kept.append({'text': text})
        write_jsonl(Path(output_path), kept)
        stats = {
            "input": len(documents),
            "dropped": dict(dropped),
            "pii_masked": pii_masked,
            "output": len(kept),
        }
        print("Filtering:", stats)
        return stats  
