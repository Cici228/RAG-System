import json
import torch
from transformers import AutoModel, AutoTokenizer
import sys

def load_documents(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data

def setup_bge_model():
    model_name = "BAAI/bge-base-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

def encode_text(model, tokenizer, text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state[:, 0]  # Use [CLS] token embedding
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].tolist()

def main():
    input_path = "documents.json"
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    
    # Load documents
    documents = load_documents(input_path)
    print(f"Loaded {len(documents)} documents")
    
    # Setup model
    model, tokenizer = setup_bge_model()
    print("Loaded BGE model")
    
    # Process documents
    output = []
    for i, doc in enumerate(documents):
        if i % 100 == 0:
            print(f"Processing document {i}/{len(documents)}")
        
        embedding = encode_text(model, tokenizer, doc["text"])
        output.append({
            "id": doc["id"],
            "text": doc["text"],
            "embedding": embedding
        })
    
    # Save results
    with open("preprocessed_documents.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved preprocessed_documents.json with {len(output)} entries")

if __name__ == "__main__":
    main()