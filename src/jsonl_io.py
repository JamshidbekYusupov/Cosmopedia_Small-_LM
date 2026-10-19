import json
def read_jsonl(path, limit=None):
    """Read a JSONL file and return a list of documents.

    limit=1000 reads only the first 1000 documents. Use it while developing:
    waiting 4 minutes to find out your regex was wrong is a bad way to learn.
    """
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            documents.append(json.loads(line))
    return documents


def write_jsonl(path, documents):
    """Write a list of documents to a JSONL file."""
    # Make sure the folder exists, otherwise open() fails on a fresh clone.
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            # ensure_ascii=False keeps real letters readable in the file
            # instead of turning them into backslash-u escape sequences.
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")