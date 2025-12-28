import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

class QueryEncoder:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        """Initialize the BGE model for encoding queries."""
        print(f"Loading BGE model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def encode(self, query_text: str) -> np.ndarray:
        """Encode a query text into a 768-dimensional vector."""
        # Format query for BGE model (same as document encoding)
        formatted_query = f"Represent this sentence for searching relevant passages: {query_text}"
        
        # Tokenize
        encoded_input = self.tokenizer(
            formatted_query,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        # Generate embedding
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Use [CLS] token embedding and normalize
            embedding = model_output.last_hidden_state[:, 0]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding[0].cpu().numpy()