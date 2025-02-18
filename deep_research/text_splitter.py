from abc import ABC, abstractmethod
from typing import List, Optional

class TextSplitter(ABC):
    """Base text splitter class that handles splitting text into chunks."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Cannot have chunk_overlap >= chunk_size")

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass

    def create_documents(self, texts: List[str]) -> List[str]:
        documents = []
        for text in texts:
            for chunk in self.split_text(text):
                documents.append(chunk)
        return documents

    def split_documents(self, documents: List[str]) -> List[str]:
        return self.create_documents(documents)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs).strip()
        return text if text else None

    def merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for d in splits:
            _len = len(d)
            if total + _len >= self.chunk_size:
                if total > self.chunk_size:
                    print(f"Created a chunk of size {total}, which is longer than the specified {self.chunk_size}")

                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)

                    while total > self.chunk_overlap or (total + _len > self.chunk_size and total > 0):
                        total -= len(current_doc[0])
                        current_doc.pop(0)

            current_doc.append(d)
            total += _len

        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

class RecursiveCharacterTextSplitter(TextSplitter):
    """Splits text recursively by different separators."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ".", ",", ">", "<", " ", ""]

    def split_text(self, text: str) -> List[str]:
        final_chunks: List[str] = []
        separator = self.separators[-1]
        for s in self.separators:
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                break
        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self.merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                other_info = self.split_text(s)
                final_chunks.extend(other_info)
        if good_splits:
            merged_text = self.merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)
        return final_chunks
