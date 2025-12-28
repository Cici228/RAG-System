# RAG-System
My implementation of a complete Retrieval-Augmented Generation (RAG) system, the same technology powering ChatGPT's web search, enterprise AI assistants, and modern search engines as final project for CS 4414.

This RAG system implementation retrieves relevant information by combining embedding-based semantic search with large language model generation. Documents are first encoded into high-dimensional (768) embedding vectors using a pretrained encoder, so that texts with similar meanings are close in vector space even if they share no exact wording. These embeddings are stored in a vector database that supports efficient similarity search; when a query arrives, it is embedded and compared against stored vectors using approximate nearest neighbor techniques to quickly identify the top-K most relevant documents. The retrieved documents are then provided as contextual input to a lightweight large language model, TinyLlama-1.1B-Chat-v0.3, a compact model that can run locally, which generates responses grounded in the retrieved content. By augmenting the LLM with semantically relevant context from the vector database, the system produces more accurate and meaningful answers than standalone generation.

## How to Run

To Start directly in interactive mode:
```
> python main.py --data preprocessed_documents.json --model tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf --interactive
```
Then ask a query from query.json

To Process one question and continue interactively:
```
> python main.py --data preprocessed_documents.json --model tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf --question "What causes squirrels to lose fur?"
```
Or switch the query

Results and analysis:
[cs 4414 HW3 (1).pdf](https://github.com/user-attachments/files/24362315/cs.4414.HW3.1.pdf)
