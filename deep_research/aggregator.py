import os

def aggregate_files(directory: str) -> list:
    """
    Aggregates content from all .txt and .md files in a directory recursively.
    Returns a list of dicts with keys: 'source' and 'content'
    """
    documents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        documents.append({"source": file_path, "content": content})
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return documents
