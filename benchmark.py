# benchmark.py
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from main import RAGSystem
import warnings
warnings.filterwarnings("ignore")

class RAGBenchmark:
    def __init__(self, data_path="preprocessed_documents.json", model_path="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"):
        """Initialize benchmark system."""
        print("Initializing RAG Benchmark System...")
        self.rag_system = RAGSystem(data_path, model_path)
        self.results = []
        
    def benchmark_component(self, component_name, func, *args):
        """Benchmark a single component."""
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        return latency, result
    
    def run_benchmark(self, questions, iterations=1):
        """Run comprehensive benchmark on multiple questions."""
        all_results = []
        
        for question in questions:
            print(f"\nBenchmarking: '{question}'")
            
            # Component 1: Query Encoding
            encode_latency, query_embedding = self.benchmark_component(
                "Query Encoding",
                self.rag_system.query_encoder.encode,
                question
            )
            
            # Component 2: Vector Search
            search_latency, (distances, indices) = self.benchmark_component(
                "Vector Search",
                self.rag_system.vector_db.search,
                query_embedding,
                3
            )
            
            # Component 3: Document Retrieval
            retrieval_latency, retrieved_docs = self.benchmark_component(
                "Document Retrieval",
                self.rag_system.vector_db.get_documents_by_indices,
                indices
            )
            
            # Component 4: Prompt Augmentation
            prompt_latency, prompt = self.benchmark_component(
                "Prompt Augmentation",
                self.rag_system.prompt_builder.build_rag_prompt,
                question,
                retrieved_docs
            )
            
            # Component 5: LLM Generation
            if self.rag_system.llm_generation:
                generation_latency, answer = self.benchmark_component(
                    "LLM Generation",
                    self.rag_system.llm_generation.generate,
                    prompt
                )
            else:
                generation_latency = 0
                answer = "LLM not available"
            
            # Total latency
            total_latency = (encode_latency + search_latency + 
                           retrieval_latency + prompt_latency + generation_latency)
            
            # Store results
            result = {
                "question": question,
                "total_ms": total_latency,
                "encode_ms": encode_latency,
                "search_ms": search_latency,
                "retrieval_ms": retrieval_latency,
                "prompt_ms": prompt_latency,
                "generation_ms": generation_latency,
                "answer_length": len(answer)
            }
            
            all_results.append(result)
            print(f"  Total: {total_latency:.2f}ms | "
                  f"Encode: {encode_latency:.2f}ms | "
                  f"Search: {search_latency:.2f}ms | "
                  f"LLM: {generation_latency:.2f}ms")
        
        return pd.DataFrame(all_results)
    
    def plot_latency_distribution(self, df):
        """Create visualization of latency distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Component-wise latency distribution
        components = ['encode_ms', 'search_ms', 'retrieval_ms', 'prompt_ms', 'generation_ms']
        component_names = ['Encoding', 'Vector Search', 'Retrieval', 'Prompt', 'LLM Generation']
        component_data = [df[col] for col in components]
        
        axes[0, 0].boxplot(component_data, labels=component_names)
        axes[0, 0].set_title('Component Latency Distribution')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Percentage breakdown
        percentages = []
        for col in components:
            percentages.append(df[col].sum() / df['total_ms'].sum() * 100)
        
        axes[0, 1].pie(percentages, labels=component_names, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Latency Percentage Breakdown')
        
        # 3. Total latency distribution
        axes[1, 0].hist(df['total_ms'], bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Total Latency Distribution')
        axes[1, 0].set_xlabel('Total Latency (ms)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(df['total_ms'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["total_ms"].mean():.2f}ms')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Correlation between answer length and generation time
        if df['generation_ms'].sum() > 0:
            axes[1, 1].scatter(df['answer_length'], df['generation_ms'], alpha=0.6)
            axes[1, 1].set_title('Answer Length vs Generation Time')
            axes[1, 1].set_xlabel('Answer Length (characters)')
            axes[1, 1].set_ylabel('Generation Time (ms)')
            
            # Add trend line
            z = np.polyfit(df['answer_length'], df['generation_ms'], 1)
            p = np.poly1d(z)
            axes[1, 1].plot(df['answer_length'], p(df['answer_length']), 
                           "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No LLM generation data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Answer Length vs Generation Time')
        
        plt.tight_layout()
        plt.savefig('latency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_report(self, df):
        """Generate a detailed performance report."""
        report = []
        report.append("=" * 80)
        report.append("RAG SYSTEM PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal Questions Analyzed: {len(df)}")
        report.append(f"Total Benchmark Time: {df['total_ms'].sum()/1000:.2f} seconds")
        report.append("\n" + "-" * 80)
        
        # Component statistics
        components = {
            'Total': 'total_ms',
            'Query Encoding': 'encode_ms',
            'Vector Search': 'search_ms',
            'Document Retrieval': 'retrieval_ms',
            'Prompt Augmentation': 'prompt_ms',
            'LLM Generation': 'generation_ms'
        }
        
        for name, col in components.items():
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                report.append(f"\n{name}:")
                report.append(f"  Mean: {mean:.2f}ms | Std: {std:.2f}ms | "
                             f"Range: [{min_val:.2f}ms, {max_val:.2f}ms]")
                if name != 'Total':
                    percentage = (df[col].sum() / df['total_ms'].sum()) * 100
                    report.append(f"  Percentage of total: {percentage:.1f}%")
        
        # Identify bottlenecks
        report.append("\n" + "-" * 80)
        report.append("BOTTLENECK ANALYSIS:")
        
        # Find the component with highest average latency
        latency_cols = ['encode_ms', 'search_ms', 'retrieval_ms', 'prompt_ms', 'generation_ms']
        avg_latencies = {col: df[col].mean() for col in latency_cols}
        bottleneck = max(avg_latencies.items(), key=lambda x: x[1])
        
        component_names = {
            'encode_ms': 'Query Encoding',
            'search_ms': 'Vector Search',
            'retrieval_ms': 'Document Retrieval',
            'prompt_ms': 'Prompt Augmentation',
            'generation_ms': 'LLM Generation'
        }
        
        report.append(f"Primary Bottleneck: {component_names[bottleneck[0]]} "
                     f"({bottleneck[1]:.2f}ms average)")
        
        # Check if LLM generation is the bottleneck
        if bottleneck[0] == 'generation_ms':
            report.append("  - LLM generation is the slowest component")
            report.append("  - Consider using a smaller model or optimizing generation parameters")
        elif bottleneck[0] == 'encode_ms':
            report.append("  - Query encoding is computationally intensive")
            report.append("  - Consider using a lighter encoder model")
        elif bottleneck[0] == 'search_ms':
            report.append("  - Vector search scales with dataset size")
            report.append("  - Consider using approximate nearest neighbor search")
        
        report.append("\n" + "-" * 80)
        report.append("PERFORMANCE OPTIMIZATION RECOMMENDATIONS:")
        report.append("1. For latency-critical applications: Focus on LLM optimization")
        report.append("2. For accuracy-critical applications: Focus on retrieval quality")
        report.append("3. Balanced approach: Optimize both retrieval and generation")
        
        return "\n".join(report)

def main():
    """Main benchmarking function."""
    # Test questions covering different domains and lengths
    test_questions = [
        "What causes squirrels to lose fur?",
        "How do I take care of backyard squirrels?",
        "What do squirrels eat in winter?",
        "Are squirrels active during night time?",
        "How do squirrels communicate with each other?",
        "What predators do squirrels have?",
        "How long do squirrels typically live?",
        "Do squirrels hibernate during winter?",
        "How do squirrels build their nests?",
        "What diseases can squirrels carry?"
    ]
    
    print("Starting RAG System Benchmark...")
    print(f"Testing with {len(test_questions)} questions")
    print("-" * 50)
    
    # Initialize benchmark system
    benchmark = RAGBenchmark()
    
    # Run benchmark
    results_df = benchmark.run_benchmark(test_questions)
    
    # Generate plots
    print("\nGenerating visualizations...")
    benchmark.plot_latency_distribution(results_df)
    
    # Generate report
    report = benchmark.generate_report(results_df)
    print(report)
    
    # Save results to CSV
    results_df.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to 'benchmark_results.csv'")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Average Total Latency: {results_df['total_ms'].mean():.2f}ms")
    print(f"Std Dev of Total Latency: {results_df['total_ms'].std():.2f}ms")
    print(f"Minimum Latency: {results_df['total_ms'].min():.2f}ms")
    print(f"Maximum Latency: {results_df['total_ms'].max():.2f}ms")
    print(f"Throughput: {len(results_df) / (results_df['total_ms'].sum() / 1000):.2f} queries/second")

if __name__ == "__main__":
    main()