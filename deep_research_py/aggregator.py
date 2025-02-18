import os

def aggregate_files(directory: str) -> str:
    """Aggregates content from all .txt and .md files in a directory recursively."""
    aggregated_content = ""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        aggregated_content += f.read() + "\n"
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return aggregated_content
