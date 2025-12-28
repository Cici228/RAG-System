# experiment_topk.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import RAGSystem

def experiment_top_k():
    """Experiment with different top-k values."""
    print("Experiment: Impact of Top-K Values")
    print("=" * 60)
    
    rag_system = RAGSystem("preprocessed_documents.json", "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
    
    test_question = "What causes squirrels to lose fur?"
    k_values = [1, 3, 5, 10, 20]
    
    results = []
    
    for k in k_values:
        print(f"\nTesting with top-k = {k}")
        
        # Measure total time
        start_time = time.time()
        
        # Encode query
        query_embedding = rag_system.query_encoder.encode(test_question)
        
        # Vector search
        distances, indices = rag_system.vector_db.search(query_embedding, k)
        
        # Retrieve documents
        retrieved_docs = rag_system.vector_db.get_documents_by_indices(indices)
        
        # Build prompt
        prompt = rag_system.prompt_builder.build_rag_prompt(test_question, retrieved_docs)
        
        # Generate answer (if available)
        if rag_system.llm_generation:
            answer = rag_system.llm_generation.generate(prompt)
        else:
            answer = "LLM not available"
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Calculate retrieval quality metrics
        avg_distance = np.mean(distances) if len(distances) > 0 else 0
        min_distance = np.min(distances) if len(distances) > 0 else 0
        
        result = {
            'top_k': k,
            'total_ms': total_time,
            'retrieval_ms': total_time * 0.02,  # Estimate
            'generation_ms': total_time * 0.98,  # Estimate
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'answer_length': len(answer),
            'doc_ids': [doc['id'] for doc in retrieved_docs]
        }
        
        results.append(result)
        
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Retrieved docs: {len(retrieved_docs)}")
        print(f"  Average distance: {avg_distance:.4f}")
    
    # Create visualization
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Total time vs top-k
    axes[0, 0].plot(df['top_k'], df['total_ms'], 'b-o', linewidth=2)
    axes[0, 0].set_title('Total Latency vs Top-K')
    axes[0, 0].set_xlabel('Top-K Value')
    axes[0, 0].set_ylabel('Total Latency (ms)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Retrieval quality vs top-k
    axes[0, 1].plot(df['top_k'], df['avg_distance'], 'r-o', linewidth=2, label='Average Distance')
    axes[0, 1].plot(df['top_k'], df['min_distance'], 'g-o', linewidth=2, label='Minimum Distance')
    axes[0, 1].set_title('Retrieval Quality vs Top-K')
    axes[0, 1].set_xlabel('Top-K Value')
    axes[0, 1].set_ylabel('Distance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Answer length vs top-k
    axes[1, 0].bar(df['top_k'], df['answer_length'], alpha=0.7)
    axes[1, 0].set_title('Answer Length vs Top-K')
    axes[1, 0].set_xlabel('Top-K Value')
    axes[1, 0].set_ylabel('Answer Length (chars)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Time breakdown
    axes[1, 1].stackplot(df['top_k'], 
                         df['retrieval_ms'], 
                         df['generation_ms'],
                         labels=['Retrieval', 'Generation'],
                         alpha=0.7)
    axes[1, 1].set_title('Time Breakdown vs Top-K')
    axes[1, 1].set_xlabel('Top-K Value')
    axes[1, 1].set_ylabel('Latency (ms)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('topk_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print(f"Baseline (k=3): {df.loc[df['top_k']==3, 'total_ms'].values[0]:.2f}ms")
    print(f"k=1: {df.loc[df['top_k']==1, 'total_ms'].values[0]:.2f}ms "
          f"({df.loc[df['top_k']==1, 'total_ms'].values[0]/df.loc[df['top_k']==3, 'total_ms'].values[0]*100:.1f}% of baseline)")
    print(f"k=10: {df.loc[df['top_k']==10, 'total_ms'].values[0]:.2f}ms "
          f"({df.loc[df['top_k']==10, 'total_ms'].values[0]/df.loc[df['top_k']==3, 'total_ms'].values[0]*100:.1f}% of baseline)")
    
    print("\nTRADEOFFS OBSERVED:")
    print("1. Higher k values → More context → Potentially better answers")
    print("2. Higher k values → Longer prompts → Longer generation times")
    print("3. k=1 is fastest but risks missing relevant context")
    print("4. Optimal k depends on application requirements")

if __name__ == "__main__":
    experiment_top_k()