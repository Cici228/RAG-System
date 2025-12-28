from llama_cpp import Llama
import argparse

class LLMGenerator:
    def __init__(self, model_path: str):
        """Initialize the LLM with TinyLlama model."""
        print(f"Loading LLM from: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context length
            n_threads=4,  # Adjust based on your CPU
            verbose=False
        )
        print("LLM loaded successfully!")
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate response using the LLM."""
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                echo=False,  # Don't echo the prompt
                stop=["</s>", "Question:", "Document:"]  # Stop sequences
            )
            
            return response['choices'][0]['text'].strip()
        
        except Exception as e:
            return f"Error generating response: {str(e)}"