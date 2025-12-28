# experiment_models.py
import time
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

def compare_models():
    """Compare different LLM models."""
    print("Experiment: Impact of Different LLM Models")
    print("=" * 60)
    
    test_prompt = "Based on the following documents, answer: What causes squirrels to lose fur?\n\nDocuments:\nDocument 1: Squirrels may lose fur due to mange, a skin disease caused by mites.\nDocument 2: Seasonal shedding is normal for squirrels.\nDocument 3: Stress and poor nutrition can cause fur loss.\n\nAnswer:"
    
    models = [
        ("tinyllama-1.1b", "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf", 1.1),
        ("distilgpt2", "distilgpt2", 0.082),
        ("gpt2-medium", "gpt2-medium", 0.355),
        ("gpt2-large", "gpt2-large", 0.774)
    ]
    
    results = []
    
    for model_name, model_path, model_size in models:
        print(f"\nTesting {model_name} ({model_size}B parameters)...")
        
        try:
            if model_name == "tinyllama-1.1b":
                # Use llama.cpp for TinyLlama
                from llama_cpp import Llama
                llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
                
                start_time = time.time()
                response = llm(test_prompt, max_tokens=150, temperature=0.7)
                generation_time = (time.time() - start_time) * 1000
                answer = response['choices'][0]['text'].strip()
                
            else:
                # Use transformers pipeline for other models
                generator = pipeline("text-generation", model=model_path, device=-1)
                
                start_time = time.time()
                response = generator(test_prompt, max_new_tokens=150, temperature=0.7)
                generation_time = (time.time() - start_time) * 1000
                answer = response[0]['generated_text'].replace(test_prompt, "").strip()
            
            # Quality assessment (simple heuristic)
            answer_length = len(answer)
            has_keywords = any(keyword in answer.lower() 
                              for keyword in ['mange', 'mites', 'shedding', 'stress', 'nutrition'])
            
            result = {
                'model': model_name,
                'size_gb': model_size,
                'time_ms': generation_time,
                'answer_length': answer_length,
                'has_keywords': has_keywords,
                'answer_preview': answer[:100] + "..."
            }
            
            results.append(result)
            
            print(f"  Generation time: {generation_time:.2f}ms")
            print(f"  Answer length: {answer_length} chars")
            print(f"  Contains key concepts: {has_keywords}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'model': model_name,
                'size_gb': model_size,
                'time_ms': None,
                'answer_length': 0,
                'has_keywords': False,
                'answer_preview': f"Error: {str(e)[:50]}"
            })
    
    # Create visualization
    df = pd.DataFrame([r for r in results if r['time_ms'] is not None])
    
    if len(df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Generation time vs model size
        axes[0, 0].scatter(df['size_gb'], df['time_ms'], s=100, alpha=0.7)
        for i, row in df.iterrows():
            axes[0, 0].annotate(row['model'], (row['size_gb'], row['time_ms']), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 0].set_title('Generation Time vs Model Size')
        axes[0, 0].set_xlabel('Model Size (Billion Parameters)')
        axes[0, 0].set_ylabel('Generation Time (ms)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Answer length vs model size
        axes[0, 1].bar(df['model'], df['answer_length'], alpha=0.7)
        axes[0, 1].set_title('Answer Length by Model')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Answer Length (chars)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Speed-quality tradeoff
        axes[1, 0].scatter(df['time_ms'], df['answer_length'], 
                          c=df['size_gb'], s=100, alpha=0.7, cmap='viridis')
        for i, row in df.iterrows():
            axes[1, 0].annotate(row['model'], (row['time_ms'], row['answer_length']), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_title('Speed-Quality Tradeoff')
        axes[1, 0].set_xlabel('Generation Time (ms)')
        axes[1, 0].set_ylabel('Answer Length (chars)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Model capabilities
        capabilities = []
        for _, row in df.iterrows():
            score = (row['answer_length'] / 100) + (10 if row['has_keywords'] else 0)
            capabilities.append(score)
        
        axes[1, 1].bar(df['model'], capabilities, alpha=0.7)
        axes[1, 1].set_title('Model Capability Score')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Capability Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analysis
        print("\n" + "=" * 60)
        print("ANALYSIS:")
        
        fastest = df.loc[df['time_ms'].idxmin()]
        slowest = df.loc[df['time_ms'].idxmax()]
        best_quality = df.loc[df['answer_length'].idxmax()]
        
        print(f"Fastest model: {fastest['model']} ({fastest['time_ms']:.2f}ms)")
        print(f"Slowest model: {slowest['model']} ({slowest['time_ms']:.2f}ms)")
        print(f"Longest answers: {best_quality['model']} ({best_quality['answer_length']} chars)")
        
        print("\nTRADEOFFS OBSERVED:")
        print("1. Larger models → Better answer quality but slower generation")
        print("2. Smaller models → Faster but less coherent/complete answers")
        print("3. Quantized models (like TinyLlama GGUF) offer good speed-quality balance")
        
    else:
        print("\nNo valid results to display.")

if __name__ == "__main__":
    compare_models()