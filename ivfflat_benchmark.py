# ivfflat_benchmark.py
import numpy as np
import faiss
import time
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class IVFFlatBenchmark:
    def __init__(self, json_path: str = "preprocessed_documents.json"):
        """Initialize benchmark with preprocessed documents."""
        print("Loading documents for IVFFlat benchmark...")
        with open(json_path, 'r') as f:
            self.documents = json.load(f)
        
        # Extract embeddings
        embeddings_list = []
        for doc in self.documents:
            embeddings_list.append(doc['embedding'])
        
        self.embeddings = np.array(embeddings_list).astype('float32')
        self.dimension = self.embeddings.shape[1]
        self.n_vectors = self.embeddings.shape[0]
        
        print(f"Loaded {self.n_vectors} vectors of dimension {self.dimension}")
    
    def build_indices(self, nlist_values: List[int] = [10, 50, 100, 200, 400]):
        """Build both Flat and IVFFlat indices."""
        print("\nBuilding indices...")
        
        # Build flat index (baseline)
        print("Building Flat index...")
        self.flat_index = faiss.IndexFlatL2(self.dimension)
        self.flat_index.add(self.embeddings)
        
        # Build IVFFlat indices with different nlist values
        self.ivfflat_indices = {}
        
        for nlist in nlist_values:
            print(f"Building IVFFlat index with nlist={nlist}...")
            
            # Create quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            
            # Create IVFFlat index
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # Train on subset of data (required for IVFFlat)
            n_train = min(100000, self.n_vectors)
            train_vectors = self.embeddings[:n_train]
            
            print(f"  Training on {n_train} vectors...")
            index.train(train_vectors)
            
            # Add all vectors
            index.add(self.embeddings)
            
            # Set nprobe (number of clusters to search)
            index.nprobe = min(10, nlist // 10)  # Search 10% of clusters
            
            self.ivfflat_indices[nlist] = index
        
        print("All indices built successfully!")
    
    def generate_test_queries(self, n_queries: int = 1000, 
                             difficulty: str = 'mixed') -> Tuple[np.ndarray, List[int]]:
        """Generate test queries with varying difficulty."""
        print(f"Generating {n_queries} {difficulty} test queries...")
        
        query_embeddings = []
        ground_truth_indices = []  # For accuracy calculation
        
        for _ in range(n_queries):
            if difficulty == 'easy':
                # Query is exactly a document (perfect match)
                idx = np.random.randint(0, self.n_vectors)
                query = self.embeddings[idx].copy()
                ground_truth = [idx]
                
            elif difficulty == 'medium':
                # Query is near a document (small noise)
                idx = np.random.randint(0, self.n_vectors)
                query = self.embeddings[idx].copy()
                noise = np.random.normal(0, 0.05, self.dimension).astype('float32')
                query = query + noise
                query = query / np.linalg.norm(query)
                ground_truth = [idx]
                
            elif difficulty == 'hard':
                # Query is between documents (larger noise)
                idx1 = np.random.randint(0, self.n_vectors)
                idx2 = np.random.randint(0, self.n_vectors)
                query = (self.embeddings[idx1] + self.embeddings[idx2]) / 2
                query = query / np.linalg.norm(query)
                ground_truth = [idx1, idx2]
                
            else:  # mixed
                # Randomly choose difficulty
                difficulties = ['easy', 'medium', 'hard']
                chosen = np.random.choice(difficulties)
                if chosen == 'easy':
                    idx = np.random.randint(0, self.n_vectors)
                    query = self.embeddings[idx].copy()
                    ground_truth = [idx]
                elif chosen == 'medium':
                    idx = np.random.randint(0, self.n_vectors)
                    query = self.embeddings[idx].copy()
                    noise = np.random.normal(0, 0.05, self.dimension).astype('float32')
                    query = query + noise
                    query = query / np.linalg.norm(query)
                    ground_truth = [idx]
                else:
                    idx1 = np.random.randint(0, self.n_vectors)
                    idx2 = np.random.randint(0, self.n_vectors)
                    query = (self.embeddings[idx1] + self.embeddings[idx2]) / 2
                    query = query / np.linalg.norm(query)
                    ground_truth = [idx1, idx2]
            
            query_embeddings.append(query)
            ground_truth_indices.append(ground_truth)
        
        return np.array(query_embeddings), ground_truth_indices
    
    def benchmark_indices(self, 
                         query_embeddings: np.ndarray,
                         ground_truth: List[List[int]],
                         k: int = 3,
                         nprobe_values: List[int] = [1, 5, 10, 20, 50]) -> pd.DataFrame:
        """Benchmark different indices with varying parameters."""
        results = []
        
        print("\nBenchmarking indices...")
        print("="*80)
        
        # Benchmark flat index
        print("\nFlat Index (exact search):")
        flat_times = []
        flat_recalls = []
        
        for query in query_embeddings:
            start_time = time.time()
            distances, indices = self.flat_index.search(query.reshape(1, -1), k)
            end_time = time.time()
            flat_times.append((end_time - start_time) * 1000)  # ms
            
            # Calculate recall
            gt_set = set(ground_truth[0])  # For simplicity, use first ground truth
            retrieved_set = set(indices[0])
            recall = len(gt_set.intersection(retrieved_set)) / len(gt_set)
            flat_recalls.append(recall)
        
        flat_avg_time = np.mean(flat_times)
        flat_avg_recall = np.mean(flat_recalls)
        
        flat_result = {
            'index_type': 'Flat',
            'nlist': 'N/A',
            'nprobe': 'N/A',
            'avg_latency_ms': flat_avg_time,
            'throughput_qps': 1000 / flat_avg_time,
            'avg_recall': flat_avg_recall,
            'build_time_ms': 0,
            'memory_mb': self.flat_index.ntotal * self.dimension * 4 / (1024 * 1024),
            'accuracy_vs_speed': flat_avg_recall / flat_avg_time * 1000
        }
        
        results.append(flat_result)
        print(f"  Avg latency: {flat_avg_time:.4f} ms")
        print(f"  Avg recall: {flat_avg_recall:.4f}")
        
        # Benchmark IVFFlat indices
        for nlist, index in self.ivfflat_indices.items():
            print(f"\nIVFFlat Index (nlist={nlist}):")
            
            for nprobe in nprobe_values:
                if nprobe > nlist:
                    continue
                    
                index.nprobe = nprobe
                
                ivfflat_times = []
                ivfflat_recalls = []
                
                for i, query in enumerate(query_embeddings):
                    start_time = time.time()
                    distances, indices = index.search(query.reshape(1, -1), k)
                    end_time = time.time()
                    ivfflat_times.append((end_time - start_time) * 1000)  # ms
                    
                    # Calculate recall
                    gt_set = set(ground_truth[i])
                    retrieved_set = set(indices[0])
                    recall = len(gt_set.intersection(retrieved_set)) / len(gt_set)
                    ivfflat_recalls.append(recall)
                
                avg_time = np.mean(ivfflat_times)
                avg_recall = np.mean(ivfflat_recalls)
                
                # Estimate build time (training + adding)
                build_time = (self.n_vectors * self.dimension * 4 / (1024 * 1024 * 1024)) * 1000  # Simplified
                
                result = {
                    'index_type': 'IVFFlat',
                    'nlist': nlist,
                    'nprobe': nprobe,
                    'avg_latency_ms': avg_time,
                    'throughput_qps': 1000 / avg_time,
                    'avg_recall': avg_recall,
                    'build_time_ms': build_time,
                    'memory_mb': index.ntotal * self.dimension * 4 / (1024 * 1024) * 1.2,  # +20% overhead
                    'accuracy_vs_speed': avg_recall / avg_time * 1000
                }
                
                results.append(result)
                print(f"  nprobe={nprobe}: {avg_time:.4f} ms, recall={avg_recall:.4f}")
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, df: pd.DataFrame):
        """Create comprehensive comparison plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Separate flat and IVFFlat results
        flat_df = df[df['index_type'] == 'Flat']
        ivfflat_df = df[df['index_type'] == 'IVFFlat']
        
        # Plot 1: Latency vs Recall (tradeoff)
        colors = plt.cm.viridis(np.linspace(0, 1, len(ivfflat_df['nlist'].unique())))
        
        for idx, nlist in enumerate(sorted(ivfflat_df['nlist'].unique())):
            subset = ivfflat_df[ivfflat_df['nlist'] == nlist]
            axes[0, 0].scatter(subset['avg_recall'], subset['avg_latency_ms'], 
                              label=f'nlist={nlist}', color=colors[idx], s=100)
            # Connect points with lines (same nlist, different nprobe)
            axes[0, 0].plot(subset['avg_recall'], subset['avg_latency_ms'], 
                           color=colors[idx], alpha=0.5)
        
        # Add flat index point
        axes[0, 0].scatter(flat_df['avg_recall'], flat_df['avg_latency_ms'], 
                          color='red', s=200, marker='*', label='Flat (exact)')
        
        axes[0, 0].set_xlabel('Average Recall')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].set_title('Accuracy-Speed Tradeoff')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Throughput comparison
        index_labels = []
        throughput_values = []
        
        # Add flat
        index_labels.append('Flat')
        throughput_values.append(flat_df['throughput_qps'].values[0])
        
        # Add best IVFFlat for each nlist
        for nlist in sorted(ivfflat_df['nlist'].unique()):
            subset = ivfflat_df[ivfflat_df['nlist'] == nlist]
            best_idx = subset['throughput_qps'].idxmax()
            index_labels.append(f'IVF-{nlist}')
            throughput_values.append(subset.loc[best_idx, 'throughput_qps'])
        
        bars = axes[0, 1].bar(index_labels, throughput_values, alpha=0.7)
        axes[0, 1].set_ylabel('Throughput (queries/sec)')
        axes[0, 1].set_title('Maximum Throughput Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars, throughput_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                           f'{value:.0f}', ha='center', va='bottom')
        
        # Plot 3: Memory usage
        memory_labels = []
        memory_values = []
        
        memory_labels.append('Flat')
        memory_values.append(flat_df['memory_mb'].values[0])
        
        for nlist in sorted(ivfflat_df['nlist'].unique()):
            subset = ivfflat_df[ivfflat_df['nlist'] == nlist]
            memory_labels.append(f'IVF-{nlist}')
            memory_values.append(subset['memory_mb'].mean())
        
        bars = axes[0, 2].bar(memory_labels, memory_values, alpha=0.7, color='orange')
        axes[0, 2].set_ylabel('Memory Usage (MB)')
        axes[0, 2].set_title('Memory Usage Comparison')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars, memory_values):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.1f}MB', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Speedup vs Flat
        speedup_labels = []
        speedup_values = []
        
        flat_latency = flat_df['avg_latency_ms'].values[0]
        
        for nlist in sorted(ivfflat_df['nlist'].unique()):
            subset = ivfflat_df[ivfflat_df['nlist'] == nlist]
            fastest_idx = subset['avg_latency_ms'].idxmin()
            speedup = flat_latency / subset.loc[fastest_idx, 'avg_latency_ms']
            speedup_labels.append(f'IVF-{nlist}')
            speedup_values.append(speedup)
        
        bars = axes[1, 0].bar(speedup_labels, speedup_values, alpha=0.7, color='green')
        axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Flat baseline')
        axes[1, 0].set_ylabel('Speedup (x times faster)')
        axes[1, 0].set_title('Speedup vs Flat Index')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars, speedup_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.1f}x', ha='center', va='bottom')
        
        # Plot 5: nprobe effect on recall (for one nlist)
        example_nlist = sorted(ivfflat_df['nlist'].unique())[2]  # Middle value
        example_df = ivfflat_df[ivfflat_df['nlist'] == example_nlist].sort_values('nprobe')
        
        axes[1, 1].plot(example_df['nprobe'], example_df['avg_recall'], 'b-o', linewidth=2)
        axes[1, 1].axhline(y=flat_df['avg_recall'].values[0], color='r', linestyle='--', 
                           label='Flat recall')
        axes[1, 1].set_xlabel('nprobe')
        axes[1, 1].set_ylabel('Average Recall')
        axes[1, 1].set_title(f'Recall vs nprobe (nlist={example_nlist})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Pareto frontier (optimal configurations)
        # Calculate pareto optimal points
        points = []
        for _, row in ivfflat_df.iterrows():
            points.append((row['avg_latency_ms'], row['avg_recall']))
        
        pareto_points = []
        for point in points:
            is_pareto = True
            for other in points:
                if (other[0] <= point[0] and other[1] >= point[1] and 
                    (other[0] < point[0] or other[1] > point[1])):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append(point)
        
        pareto_points.sort(key=lambda x: x[0])  # Sort by latency
        
        # Plot all points
        scatter = axes[1, 2].scatter(ivfflat_df['avg_latency_ms'], ivfflat_df['avg_recall'],
                                    c=ivfflat_df['nlist'], cmap='viridis', alpha=0.6, s=50)
        
        # Plot pareto frontier
        if len(pareto_points) >= 2:
            pareto_x, pareto_y = zip(*pareto_points)
            axes[1, 2].plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Frontier')
        
        # Add flat point
        axes[1, 2].scatter(flat_df['avg_latency_ms'], flat_df['avg_recall'],
                          color='red', s=200, marker='*', label='Flat (exact)')
        
        axes[1, 2].set_xlabel('Latency (ms)')
        axes[1, 2].set_ylabel('Recall')
        axes[1, 2].set_title('Pareto Frontier of IVFFlat Configurations')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add colorbar for nlist
        plt.colorbar(scatter, ax=axes[1, 2], label='nlist')
        
        plt.tight_layout()
        plt.savefig('ivfflat_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def analyze_results(self, df: pd.DataFrame):
        """Analyze and explain the benchmarking results."""
        print("\n" + "="*80)
        print("IVFFLAT PERFORMANCE ANALYSIS")
        print("="*80)
        
        flat_df = df[df['index_type'] == 'Flat']
        ivfflat_df = df[df['index_type'] == 'IVFFlat']
        
        # Find best configurations
        best_speed = ivfflat_df.loc[ivfflat_df['avg_latency_ms'].idxmin()]
        best_accuracy = ivfflat_df.loc[ivfflat_df['avg_recall'].idxmax()]
        best_tradeoff = ivfflat_df.loc[ivfflat_df['accuracy_vs_speed'].idxmax()]
        
        print(f"\nFlat Index (Baseline):")
        print(f"  Latency: {flat_df['avg_latency_ms'].values[0]:.4f} ms")
        print(f"  Recall: {flat_df['avg_recall'].values[0]:.4f}")
        print(f"  Throughput: {flat_df['throughput_qps'].values[0]:.0f} queries/sec")
        
        print(f"\nBest Speed (IVFFlat):")
        print(f"  Configuration: nlist={best_speed['nlist']}, nprobe={best_speed['nprobe']}")
        print(f"  Latency: {best_speed['avg_latency_ms']:.4f} ms ({best_speed['avg_latency_ms']/flat_df['avg_latency_ms'].values[0]*100:.1f}% of flat)")
        print(f"  Recall: {best_speed['avg_recall']:.4f} ({best_speed['avg_recall']/flat_df['avg_recall'].values[0]*100:.1f}% of flat)")
        print(f"  Speedup: {flat_df['avg_latency_ms'].values[0]/best_speed['avg_latency_ms']:.1f}x")
        
        print(f"\nBest Accuracy (IVFFlat):")
        print(f"  Configuration: nlist={best_accuracy['nlist']}, nprobe={best_accuracy['nprobe']}")
        print(f"  Recall: {best_accuracy['avg_recall']:.4f} ({best_accuracy['avg_recall']/flat_df['avg_recall'].values[0]*100:.1f}% of flat)")
        print(f"  Latency: {best_accuracy['avg_latency_ms']:.4f} ms")
        
        print(f"\nBest Tradeoff (IVFFlat):")
        print(f"  Configuration: nlist={best_tradeoff['nlist']}, nprobe={best_tradeoff['nprobe']}")
        print(f"  Accuracy/Speed ratio: {best_tradeoff['accuracy_vs_speed']:.2f}")
        print(f"  Latency: {best_tradeoff['avg_latency_ms']:.4f} ms")
        print(f"  Recall: {best_tradeoff['avg_recall']:.4f}")
        
        print("\n" + "-"*80)
        print("HOW IVFFLAT WORKS:")
        print("-"*80)
        
        print("\n1. CLUSTERING PHASE:")
        print("   - Documents are partitioned into nlist clusters using k-means")
        print("   - Each cluster has a centroid representing the 'average' vector")
        print("   - Training phase: O(n * d * nlist * iterations)")
        
        print("\n2. SEARCH PHASE:")
        print("   - Query is compared to cluster centroids (nlist comparisons)")
        print("   - Only nprobe most promising clusters are searched deeply")
        print("   - Within each cluster, exact search on cluster members")
        print("   - Complexity: O(nlist + nprobe * (n/nlist)) vs O(n) for flat")
        
        print("\n3. KEY PARAMETERS:")
        print("   - nlist: Number of clusters (higher = more precise, slower)")
        print("   - nprobe: Number of clusters to search (higher = more accurate, slower)")
        print("   - Tradeoff: nprobe ≈ sqrt(nlist) often works well")
        
        print("\n" + "-"*80)
        print("WHEN TO USE IVFFLAT:")
        print("-"*80)
        
        print("\n✅ Use IVFFlat when:")
        print("   - Dataset has > 100,000 vectors")
        print("   - Latency requirements are strict (< 10ms)")
        print("   - Can tolerate small accuracy loss (5-10%)")
        print("   - Have memory for index structure (20-30% overhead)")
        
        print("\n❌ Use Flat when:")
        print("   - Dataset is small (< 10,000 vectors)")
        print("   - 100% accuracy is required")
        print("   - Memory is extremely constrained")
        print("   - Queries can be batched effectively")
        
        print("\n" + "-"*80)
        print("RECOMMENDED CONFIGURATIONS:")
        print("-"*80)
        
        # Calculate optimal configurations for different scenarios
        n_vectors = self.n_vectors
        
        print(f"\nFor {n_vectors:,} vectors:")
        print(f"  1. Balanced: nlist = {int(np.sqrt(n_vectors))}, nprobe = {int(np.sqrt(int(np.sqrt(n_vectors))))}")
        print(f"  2. Fast: nlist = {int(n_vectors ** 0.4)}, nprobe = 1-5")
        print(f"  3. Accurate: nlist = {int(n_vectors ** 0.6)}, nprobe = {int(n_vectors ** 0.3)}")
        
        print("\nRule of thumb:")
        print("  - nlist = sqrt(n) for balanced performance")
        print("  - nprobe = sqrt(nlist) for good accuracy")
        print("  - Memory overhead: 20-30% of flat index")

def main():
    """Main IVFFlat benchmarking function."""
    print("IVFFlat vs Flat Index Benchmark")
    print("="*60)
    
    # Initialize benchmark
    benchmark = IVFFlatBenchmark()
    
    # Build indices with different nlist values
    nlist_values = [10, 50, 100, 200, 400]
    benchmark.build_indices(nlist_values)
    
    # Generate test queries
    n_queries = 500
    queries, ground_truth = benchmark.generate_test_queries(n_queries, 'mixed')
    
    # Test different nprobe values
    nprobe_values = [1, 5, 10, 20, 50]
    
    # Run benchmark
    results_df = benchmark.benchmark_indices(queries, ground_truth, k=3, nprobe_values=nprobe_values)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    benchmark.plot_comparison(results_df)
    
    # Analyze results
    benchmark.analyze_results(results_df)
    
    # Save detailed results
    results_df.to_csv('ivfflat_benchmark_results.csv', index=False)
    print("\nDetailed results saved to 'ivfflat_benchmark_results.csv'")

if __name__ == "__main__":
    main()