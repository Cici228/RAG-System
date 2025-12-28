from typing import List, Dict

class PromptBuilder:
    @staticmethod
    def build_rag_prompt(question: str, documents: List[Dict]) -> str:
        """Build a RAG prompt combining question and retrieved documents."""
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i} (ID: {doc['id']}): {doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create the augmented prompt
        prompt = f"""Based on the following documents, please answer the question.

Documents:
{context}

Question: {question}

Answer: """
        
        return prompt
    
    @staticmethod
    def build_simple_prompt(question: str, documents: List[Dict]) -> str:
        """Build a simpler prompt format."""
        context = " ".join([doc['text'] for doc in documents])
        return f"Question: {question}\n\nContext: {context}\n\nAnswer:"