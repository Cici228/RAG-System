# batch_vector_search.py
import numpy as np
import faiss
import time
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import pandas as pd

class BatchVectorSearchBenchmark:
    def __init__(self, json_path: str = "preprocessed_documents.json"):
        """Initialize benchmark with preprocessed documents."""
        print("Loading documents...")
        with open(json_path, 'r') as f:
            self.documents = json.load(f)
        
        # Extract embeddings
        embeddings_list = []
        for doc in self.documents:
            embeddings_list.append(doc['embedding'])
        
        self.embeddings = np.array(embeddings_list).astype('float32')
        self.dimension = self.embeddings.shape[1]
        self.n_vectors = self.embeddings.shape[0]
        
        # Build flat index
        self.flat_index = faiss.IndexFlatL2(self.dimension)
        self.flat_index.add(self.embeddings)
        
        print(f"Loaded {self.n_vectors} vectors of dimension {self.dimension}")
    
    def generate_test_queries(self, n_queries: int = 1000) -> np.ndarray:
        """Generate synthetic test queries."""
        print(f"Generating {n_queries} test queries...")
        # Generate queries similar to document embeddings
        query_embeddings = []
        for _ in range(n_queries):
            # Randomly select a document and add noise
            idx = np.random.randint(0, self.n_vectors)
            query = self.embeddings[idx].copy()
            # Add some Gaussian noise
            noise = np.random.normal(0, 0.1, self.dimension).astype('float32')
            query = query + noise
            # Normalize
            query = query / np.linalg.norm(query)
            query_embeddings.append(query)
        
        return np.array(query_embeddings)
    
    def benchmark_batch_search(self, 
                              query_embeddings: np.ndarray, 
                              batch_sizes: List[int],
                              k: int = 3) -> pd.DataFrame:
        """Benchmark vector search with different batch sizes."""
        results = []
        
        print("\nBenchmarking batch search performance...")
        print("=" * 60)
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Prepare queries in batches
            n_queries = len(query_embeddings)
            n_batches = (n_queries + batch_size - 1) // batch_size
            
            total_latency = 0
            total_results = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_queries)
                
                batch_queries = query_embeddings[start_idx:end_idx]
                
                # Measure search time
                start_time = time.time()
                distances, indices = self.flat_index.search(batch_queries, k)
                end_time = time.time()
                
                batch_latency = (end_time - start_time) * 1000  # ms
                total_latency += batch_latency
                total_results += len(batch_queries)
            
            # Calculate metrics
            avg_latency_per_query = total_latency / n_queries
            avg_latency_per_batch = total_latency / n_batches
            throughput = (n_queries / (total_latency / 1000))  # queries per second
            
            # Memory usage estimation
            batch_memory = batch_size * self.dimension * 4  # 4 bytes per float32
            
            result = {
                'batch_size': batch_size,
                'avg_latency_per_query_ms': avg_latency_per_query,
                'avg_latency_per_batch_ms': avg_latency_per_batch,
                'throughput_qps': throughput,
                'n_batches': n_batches,
                'batch_memory_mb': batch_memory / (1024 * 1024),
                'total_latency_ms': total_latency,
                'n_queries': n_queries
            }
            
            results.append(result)
            
            print(f"  Avg latency per query: {avg_latency_per_query:.4f} ms")
            print(f"  Throughput: {throughput:.2f} queries/sec")
            print(f"  Batch memory: {batch_memory/(1024*1024):.2f} MB")
        
        return pd.DataFrame(results)
    
    def plot_batch_performance(self, df: pd.DataFrame):
        """Create comprehensive visualization of batch performance."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Latency per query vs batch size
        axes[0, 0].plot(df['batch_size'], df['avg_latency_per_query_ms'], 'b-o', linewidth=2)
        axes[0, 0].set_xscale('log', base=2)
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Latency per Query (ms)')
        axes[0, 0].set_title('Query Latency vs Batch Size')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add annotations
        for i, row in df.iterrows():
            axes[0, 0].annotate(f"{row['avg_latency_per_query_ms']:.2f}ms", 
                               (row['batch_size'], row['avg_latency_per_query_ms']),
                               textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot 2: Throughput vs batch size
        axes[0, 1].plot(df['batch_size'], df['throughput_qps'], 'g-s', linewidth=2)
        axes[0, 1].set_xscale('log', base=2)
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Throughput (queries/sec)')
        axes[0, 1].set_title('Throughput vs Batch Size')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calculate and show speedup
        baseline_qps = df.loc[df['batch_size'] == 1, 'throughput_qps'].values[0]
        max_qps = df['throughput_qps'].max()
        max_qps_batch = df.loc[df['throughput_qps'].idxmax(), 'batch_size']
        speedup = max_qps / baseline_qps
        
        axes[0, 1].annotate(f"Speedup: {speedup:.1f}x\nat batch size {max_qps_batch}", 
                           (max_qps_batch, max_qps),
                           xytext=(10, -30), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->'))
        
        # Plot 3: Batch latency vs batch size
        axes[0, 2].plot(df['batch_size'], df['avg_latency_per_batch_ms'], 'r-^', linewidth=2)
        axes[0, 2].set_xscale('log', base=2)
        axes[0, 2].set_xlabel('Batch Size')
        axes[0, 2].set_ylabel('Latency per Batch (ms)')
        axes[0, 2].set_title('Batch Latency vs Batch Size')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Memory usage vs batch size
        axes[1, 0].plot(df['batch_size'], df['batch_memory_mb'], 'm-D', linewidth=2)
        axes[1, 0].set_xscale('log', base=2)
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage vs Batch Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Latency-Throughput tradeoff
        axes[1, 1].scatter(df['avg_latency_per_query_ms'], df['throughput_qps'], 
                          c=df['batch_size'], s=100, cmap='viridis')
        axes[1, 1].set_xlabel('Latency per Query (ms)')
        axes[1, 1].set_ylabel('Throughput (queries/sec)')
        axes[1, 1].set_title('Latency-Throughput Tradeoff')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add batch size labels
        for i, row in df.iterrows():
            axes[1, 1].annotate(f"B={row['batch_size']}", 
                               (row['avg_latency_per_query_ms'], row['throughput_qps']),
                               textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        
        # Plot 6: Efficiency (Throughput per memory unit)
        df['efficiency'] = df['throughput_qps'] / df['batch_memory_mb']
        axes[1, 2].plot(df['batch_size'], df['efficiency'], 'c-p', linewidth=2)
        axes[1, 2].set_xscale('log', base=2)
        axes[1, 2].set_xlabel('Batch Size')
        axes[1, 2].set_ylabel('Throughput per MB (qps/MB)')
        axes[1, 2].set_title('Memory Efficiency vs Batch Size')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Find optimal batch size
        optimal_idx = df['efficiency'].idxmax()
        optimal_batch = df.loc[optimal_idx, 'batch_size']
        optimal_efficiency = df.loc[optimal_idx, 'efficiency']
        
        axes[1, 2].annotate(f"Optimal: B={optimal_batch}\nEff={optimal_efficiency:.1f} qps/MB", 
                           (optimal_batch, optimal_efficiency),
                           xytext=(10, 10), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->'))
        
        plt.tight_layout()
        plt.savefig('batch_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def analyze_results(self, df: pd.DataFrame):
        """Analyze and explain the benchmarking results."""
        print("\n" + "="*80)
        print("BATCHING PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Find key metrics
        baseline = df.loc[df['batch_size'] == 1]
        optimal_throughput = df.loc[df['throughput_qps'].idxmax()]
        optimal_efficiency = df.loc[df['efficiency'].idxmax()]
        
        print(f"\nBaseline (batch size 1):")
        print(f"  Latency: {baseline['avg_latency_per_query_ms'].values[0]:.4f} ms/query")
        print(f"  Throughput: {baseline['throughput_qps'].values[0]:.2f} queries/sec")
        
        print(f"\nOptimal Throughput (batch size {optimal_throughput['batch_size']}):")
        print(f"  Latency: {optimal_throughput['avg_latency_per_query_ms']:.4f} ms/query")
        print(f"  Throughput: {optimal_throughput['throughput_qps']:.2f} queries/sec")
        print(f"  Speedup: {optimal_throughput['throughput_qps']/baseline['throughput_qps'].values[0]:.2f}x")
        
        print(f"\nOptimal Efficiency (batch size {optimal_efficiency['batch_size']}):")
        print(f"  Throughput per MB: {optimal_efficiency['efficiency']:.2f} qps/MB")
        
        print("\n" + "-"*80)
        print("OBSERVATIONS AND EXPLANATIONS:")
        print("-"*80)
        
        print("\n1. AMORTIZATION OF OVERHEAD:")
        print("   - FAISS has fixed overhead per search call (memory allocation, function calls)")
        print("   - Batching amortizes this overhead across multiple queries")
        print(f"   - Overhead reduction: From {baseline['avg_latency_per_query_ms'].values[0]:.2f}ms to {df['avg_latency_per_query_ms'].min():.2f}ms")
        
        print("\n2. CACHE EFFICIENCY:")
        print("   - Modern CPUs have multi-level caches (L1, L2, L3)")
        print("   - Processing multiple queries together improves cache locality")
        print("   - Vectors stay in cache between queries, reducing memory access latency")
        
        print("\n3. VECTORIZED OPERATIONS:")
        print("   - FAISS uses SIMD (Single Instruction Multiple Data) instructions")
        print("   - Batch processing enables better utilization of SIMD units")
        print("   - Parallel computation of distances for multiple queries")
        
        print("\n4. MEMORY BANDWIDTH UTILIZATION:")
        print("   - Memory access has high latency (~100ns)")
        print("   - Batching enables more efficient use of memory bandwidth")
        print("   - Reduces per-query memory access overhead")
        
        print("\n5. DIMINISHING RETURNS:")
        print("   - Beyond optimal batch size, benefits decrease")
        print("   - Memory pressure increases (cache thrashing)")
        print("   - CPU pipeline stalls due to resource contention")
        
        print("\n" + "-"*80)
        print("RECOMMENDATIONS:")
        print("-"*80)
        
        print("\nFor latency-sensitive applications:")
        print(f"  - Use batch size {optimal_efficiency['batch_size']} for best balance")
        print(f"  - Expected latency: {optimal_efficiency['avg_latency_per_query_ms']:.2f} ms")
        
        print("\nFor throughput-sensitive applications:")
        print(f"  - Use batch size {optimal_throughput['batch_size']} for max throughput")
        print(f"  - Expected throughput: {optimal_throughput['throughput_qps']:.0f} qps")
        
        print("\nFor memory-constrained systems:")
        print(f"  - Use batch size ≤ {df.loc[df['batch_memory_mb'] <= 10, 'batch_size'].max()}")
        print("  - Keep batch memory under 10 MB to avoid swapping")

def main():
    """Main benchmarking function."""
    print("Vector Search Batching Optimization Benchmark")
    print("="*60)
    
    # Initialize benchmark
    benchmark = BatchVectorSearchBenchmark()
    
    # Generate test queries
    n_queries = 1000
    queries = benchmark.generate_test_queries(n_queries)
    
    # Test batch sizes (powers of 2)
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    
    # Run benchmark
    results_df = benchmark.benchmark_batch_search(queries, batch_sizes, k=3)
    
    # Generate plots
    print("\nGenerating performance analysis plots...")
    benchmark.plot_batch_performance(results_df)
    
    # Analyze results
    benchmark.analyze_results(results_df)
    
    # Save detailed results
    results_df.to_csv('batch_benchmark_results.csv', index=False)
    print("\nDetailed results saved to 'batch_benchmark_results.csv'")

if __name__ == "__main__":
    main()