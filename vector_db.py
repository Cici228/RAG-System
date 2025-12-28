import json
import numpy as np
import faiss
from typing import List, Dict, Tuple

class VectorDB:
    def __init__(self, json_path: str):
        """Initialize VectorDB with preprocessed documents."""
        self.documents = []
        self.embeddings = None
        self.index = None
        self.doc_id_to_index = {}
        self.load_documents(json_path)
        self.build_index()
    
    def load_documents(self, json_path: str):
        """Load documents from preprocessed JSON file."""
        print(f"Loading documents from {json_path}...")
        with open(json_path, 'r') as f:
            self.documents = json.load(f)
        
        # Build mapping from document ID to index
        self.doc_id_to_index = {}
        for idx, doc in enumerate(self.documents):
            self.doc_id_to_index[doc['id']] = idx
        
        print(f"Loaded {len(self.documents)} documents.")
    
    def build_index(self):
        """Build FAISS index from loaded documents."""
        if not self.documents:
            raise ValueError("No documents loaded.")
        
        # Extract embeddings
        embeddings_list = []
        for doc in self.documents:
            embeddings_list.append(doc['embedding'])
        
        self.embeddings = np.array(embeddings_list).astype('float32')
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        print(f"Built FAISS index with {self.index.ntotal} vectors of dimension {dimension}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar documents using a query embedding."""
        if self.index is None:
            raise ValueError("Index not built.")
        
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_array, top_k)
        
        return distances[0], indices[0]
    
    def get_documents_by_indices(self, indices: List[int]) -> List[Dict]:
        """Retrieve documents by their indices."""
        retrieved_docs = []
        for idx in indices:
            if 0 <= idx < len(self.documents):
                retrieved_docs.append(self.documents[idx])
        return retrieved_docs
    
    def get_document_by_id(self, doc_id: int) -> Dict:
        """Get document by its original ID."""
        if doc_id not in self.doc_id_to_index:
            raise ValueError(f"Document with ID {doc_id} not found")
        return self.documents[self.doc_id_to_index[doc_id]]