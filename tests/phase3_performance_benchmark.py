"""
Phase 3 Advanced Performance Testing and Benchmarking Suite
Comprehensive stress testing and scalability analysis for optimized data structures

This module provides:
1. Stress testing with large datasets
2. Scalability analysis across multiple dimensions
3. Memory usage profiling
4. Concurrent access benchmarking
5. Performance regression detection
"""

import sys
import os
import time
import random
import threading
import psutil
import gc
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_structures.hash_table import UserItemHashTable
from data_structures.similarity_graph import ProductSimilarityGraph
from data_structures.behavior_tree import UserBehaviorTree
from data_structures.category_tree import CategoryHierarchyTree
from data_structures.optimized_hash_table import OptimizedUserItemHashTable
from data_structures.optimized_similarity_graph import OptimizedProductSimilarityGraph
from recommendation_engine import RecommendationEngine


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for Phase 3 optimization analysis.
    """
    
    def __init__(self, output_dir="performance_results"):
        """Initialize benchmarking suite."""
        self.output_dir = output_dir
        self.results = {}
        self.process = psutil.Process()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_comprehensive_benchmark(self):
        """Run complete benchmark suite."""
        print("=" * 80)
        print("PHASE 3 COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("=" * 80)
        
        # Individual data structure benchmarks
        self.benchmark_hash_table_optimization()
        self.benchmark_graph_optimization()
        self.benchmark_scalability_analysis()
        self.benchmark_concurrent_access()
        self.benchmark_memory_efficiency()
        
        # System integration benchmarks
        self.benchmark_recommendation_engine()
        
        # Generate reports
        self.generate_performance_report()
        self.generate_visualizations()
        
        print("\\nBenchmark complete! Results saved to:", self.output_dir)
    
    def benchmark_hash_table_optimization(self):
        """Benchmark hash table optimizations."""
        print("\\n" + "="*60)
        print("HASH TABLE OPTIMIZATION BENCHMARK")
        print("="*60)
        
        sizes = [1000, 5000, 10000, 50000, 100000]
        original_results = []
        optimized_results = []
        
        for size in sizes:
            print(f"\\nTesting with {size} entries...")
            
            # Generate test data
            test_data = []
            for i in range(size):
                user_id = f"user_{i % (size // 10):05d}"
                item_id = f"item_{random.randint(0, size):05d}"
                data = {
                    "rating": random.uniform(1.0, 5.0),
                    "timestamp": datetime.now().isoformat(),
                    "action": random.choice(["view", "purchase", "cart_add"])
                }
                test_data.append((user_id, item_id, data))
            
            # Test original hash table
            original_stats = self._test_hash_table(UserItemHashTable(size//10), test_data, "Original")
            original_results.append((size, original_stats))
            
            # Test optimized hash table
            optimized_stats = self._test_hash_table(OptimizedUserItemHashTable(size//10), test_data, "Optimized")
            optimized_results.append((size, optimized_stats))
            
            # Memory usage comparison
            print(f"  Memory improvement: {(original_stats['memory_mb'] - optimized_stats['memory_mb']):.2f} MB")
            print(f"  Insertion speedup: {original_stats['insert_time'] / optimized_stats['insert_time']:.2f}x")
            print(f"  Lookup speedup: {original_stats['lookup_time'] / optimized_stats['lookup_time']:.2f}x")
        
        self.results['hash_table_optimization'] = {
            'original': original_results,
            'optimized': optimized_results
        }
    
    def _test_hash_table(self, hash_table, test_data, table_type):
        """Test hash table performance."""
        # Measure memory before
        gc.collect()
        memory_before = self.process.memory_info().rss / 1024 / 1024
        
        # Insertion benchmark
        start_time = time.time()
        if hasattr(hash_table, 'batch_insert'):
            hash_table.batch_insert(test_data)
        else:
            for user_id, item_id, data in test_data:
                hash_table.insert(user_id, item_id, data)
        insert_time = time.time() - start_time
        
        # Lookup benchmark
        lookup_sample = random.sample(test_data, min(1000, len(test_data)))
        start_time = time.time()
        for user_id, item_id, _ in lookup_sample:
            hash_table.get(user_id, item_id)
        lookup_time = time.time() - start_time
        
        # Memory after
        memory_after = self.process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        # Get performance statistics
        if hasattr(hash_table, 'get_performance_statistics'):
            perf_stats = hash_table.get_performance_statistics()
        else:
            perf_stats = hash_table.get_statistics()
        
        print(f"    {table_type} Hash Table:")
        print(f"      Insert time: {insert_time:.4f}s ({len(test_data)/insert_time:.0f} ops/sec)")
        print(f"      Lookup time: {lookup_time:.4f}s ({len(lookup_sample)/lookup_time:.0f} ops/sec)")
        print(f"      Memory used: {memory_used:.2f} MB")
        if 'cache_hit_rate' in perf_stats:
            print(f"      Cache hit rate: {perf_stats['cache_hit_rate']:.3f}")
        
        return {
            'insert_time': insert_time,
            'lookup_time': lookup_time,
            'memory_mb': memory_used,
            'performance_stats': perf_stats
        }
    
    def benchmark_graph_optimization(self):
        """Benchmark graph optimization improvements."""
        print("\\n" + "="*60)
        print("SIMILARITY GRAPH OPTIMIZATION BENCHMARK")
        print("="*60)
        
        sizes = [100, 500, 1000, 2000, 5000]
        original_results = []
        optimized_results = []
        
        for num_products in sizes:
            print(f"\\nTesting with {num_products} products...")
            
            # Generate test data
            products = [f"product_{i:05d}" for i in range(num_products)]
            similarities = []
            for _ in range(min(num_products * 5, 10000)):  # Limit for reasonable test time
                p1 = random.choice(products)
                p2 = random.choice(products)
                if p1 != p2:
                    similarity = random.uniform(0.0, 1.0)
                    similarities.append((p1, p2, similarity))
            
            # Test original graph
            original_stats = self._test_similarity_graph(ProductSimilarityGraph(), products, similarities, "Original")
            original_results.append((num_products, original_stats))
            
            # Test optimized graph
            optimized_stats = self._test_similarity_graph(OptimizedProductSimilarityGraph(), products, similarities, "Optimized")
            optimized_results.append((num_products, optimized_stats))
            
            # Improvement metrics
            print(f"  Memory improvement: {(original_stats['memory_mb'] - optimized_stats['memory_mb']):.2f} MB")
            print(f"  Construction speedup: {original_stats['construction_time'] / optimized_stats['construction_time']:.2f}x")
            print(f"  Query speedup: {original_stats['query_time'] / optimized_stats['query_time']:.2f}x")
        
        self.results['graph_optimization'] = {
            'original': original_results,
            'optimized': optimized_results
        }
    
    def _test_similarity_graph(self, graph, products, similarities, graph_type):
        """Test similarity graph performance."""
        # Measure memory before
        gc.collect()
        memory_before = self.process.memory_info().rss / 1024 / 1024
        
        # Construction benchmark
        start_time = time.time()
        for product in products:
            if hasattr(graph, 'add_product'):
                graph.add_product(product)
        
        for p1, p2, similarity in similarities:
            graph.add_similarity_edge(p1, p2, similarity)
        construction_time = time.time() - start_time
        
        # Query benchmark
        start_time = time.time()
        for _ in range(100):
            test_product = random.choice(products)
            graph.get_similar_products(test_product)
        query_time = time.time() - start_time
        
        # Memory after
        memory_after = self.process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        print(f"    {graph_type} Graph:")
        print(f"      Construction: {construction_time:.4f}s")
        print(f"      Query time: {query_time:.4f}s (100 queries)")
        print(f"      Memory used: {memory_used:.2f} MB")
        
        return {
            'construction_time': construction_time,
            'query_time': query_time,
            'memory_mb': memory_used
        }
    
    def benchmark_scalability_analysis(self):
        """Analyze scalability characteristics."""
        print("\\n" + "="*60)
        print("SCALABILITY ANALYSIS")
        print("="*60)
        
        # Test different dataset dimensions
        scalability_tests = [
            ("Users", [1000, 5000, 10000, 25000, 50000], "users"),
            ("Items", [1000, 5000, 10000, 25000, 50000], "items"),
            ("Interactions per user", [5, 10, 25, 50, 100], "interactions_per_user")
        ]
        
        for test_name, test_values, test_type in scalability_tests:
            print(f"\\n{test_name} Scalability Test:")
            results = []
            
            for value in test_values:
                print(f"  Testing {test_name.lower()}: {value}")
                
                # Generate test scenario
                if test_type == "users":
                    num_users, num_items, interactions_per_user = value, 1000, 10
                elif test_type == "items":
                    num_users, num_items, interactions_per_user = 1000, value, 10
                else:  # interactions_per_user
                    num_users, num_items, interactions_per_user = 1000, 1000, value
                
                # Test recommendation engine performance
                engine_stats = self._test_recommendation_engine_scalability(
                    num_users, num_items, interactions_per_user)
                results.append((value, engine_stats))
                
                print(f"    Processing time: {engine_stats['total_time']:.4f}s")
                print(f"    Memory usage: {engine_stats['memory_mb']:.2f} MB")
            
            self.results[f'scalability_{test_type}'] = results
    
    def _test_recommendation_engine_scalability(self, num_users, num_items, interactions_per_user):
        """Test recommendation engine with specific parameters."""
        gc.collect()
        memory_before = self.process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        
        # Create recommendation engine
        engine = RecommendationEngine()
        
        # Generate interactions
        for user_i in range(num_users):
            user_id = f"user_{user_i:05d}"
            for _ in range(interactions_per_user):
                item_id = f"item_{random.randint(0, num_items-1):05d}"
                rating = random.uniform(1.0, 5.0)
                action = random.choice(["view", "purchase", "cart_add"])
                engine.add_user_interaction(user_id, item_id, rating, action)
        
        # Test recommendation generation
        sample_users = random.sample([f"user_{i:05d}" for i in range(num_users)], 
                                   min(100, num_users))
        for user_id in sample_users:
            engine.get_hybrid_recommendations(user_id, 5)
        
        total_time = time.time() - start_time
        memory_after = self.process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        return {
            'total_time': total_time,
            'memory_mb': memory_used,
            'num_users': num_users,
            'num_items': num_items,
            'interactions_per_user': interactions_per_user
        }
    
    def benchmark_concurrent_access(self):
        """Benchmark concurrent access performance."""
        print("\\n" + "="*60)
        print("CONCURRENT ACCESS BENCHMARK")
        print("="*60)
        
        thread_counts = [1, 2, 4, 8, 16]
        results = []
        
        for num_threads in thread_counts:
            print(f"\\nTesting with {num_threads} threads...")
            
            # Test optimized hash table concurrent access
            hash_table = OptimizedUserItemHashTable(1000, enable_cache=True)
            
            # Pre-populate with some data
            for i in range(1000):
                user_id = f"user_{i:03d}"
                item_id = f"item_{i:03d}"
                data = {"rating": 4.0, "action": "purchase"}
                hash_table.insert(user_id, item_id, data)
            
            # Concurrent operations test
            operations_per_thread = 1000
            start_time = time.time()
            
            def worker_thread(thread_id):
                ops_completed = 0
                for i in range(operations_per_thread):
                    # Mix of reads and writes
                    if i % 3 == 0:  # 33% writes
                        user_id = f"user_{thread_id}_{i:03d}"
                        item_id = f"item_{i:03d}"
                        data = {"rating": random.uniform(1.0, 5.0), "action": "purchase"}
                        hash_table.insert(user_id, item_id, data)
                    else:  # 67% reads
                        user_id = f"user_{random.randint(0, 999):03d}"
                        item_id = f"item_{random.randint(0, 999):03d}"
                        hash_table.get(user_id, item_id)
                    ops_completed += 1
                return ops_completed
            
            # Run concurrent threads
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
                total_ops = sum(future.result() for future in as_completed(futures))
            
            total_time = time.time() - start_time
            ops_per_second = total_ops / total_time
            
            print(f"  Total operations: {total_ops}")
            print(f"  Time: {total_time:.4f}s")
            print(f"  Throughput: {ops_per_second:.0f} ops/sec")
            
            results.append({
                'num_threads': num_threads,
                'total_ops': total_ops,
                'total_time': total_time,
                'ops_per_second': ops_per_second
            })
        
        self.results['concurrent_access'] = results
    
    def benchmark_memory_efficiency(self):
        """Analyze memory usage patterns and efficiency."""
        print("\\n" + "="*60)
        print("MEMORY EFFICIENCY ANALYSIS")
        print("="*60)
        
        # Test memory growth patterns
        sizes = [1000, 5000, 10000, 25000, 50000]
        memory_results = {}
        
        for data_structure in ["hash_table", "similarity_graph"]:
            print(f"\\nTesting {data_structure.replace('_', ' ').title()}:")
            
            memory_usage = []
            for size in sizes:
                gc.collect()  # Clean up before measurement
                memory_before = self.process.memory_info().rss / 1024 / 1024
                
                if data_structure == "hash_table":
                    # Test hash table memory usage
                    hash_table = OptimizedUserItemHashTable(size//10)
                    for i in range(size):
                        user_id = f"user_{i % 100:03d}"
                        item_id = f"item_{i:05d}"
                        data = {"rating": 4.0, "timestamp": "2023-01-01"}
                        hash_table.insert(user_id, item_id, data)
                
                else:  # similarity_graph
                    # Test graph memory usage
                    graph = OptimizedProductSimilarityGraph()
                    products = [f"product_{i:05d}" for i in range(size//10)]
                    for product in products:
                        graph.add_product(product)
                    
                    # Add edges
                    for _ in range(size):
                        p1 = random.choice(products)
                        p2 = random.choice(products)
                        similarity = random.uniform(0.3, 1.0)
                        graph.add_similarity_edge(p1, p2, similarity)
                
                memory_after = self.process.memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before
                memory_usage.append((size, memory_used))
                
                print(f"  Size {size}: {memory_used:.2f} MB")
            
            memory_results[data_structure] = memory_usage
        
        self.results['memory_efficiency'] = memory_results
    
    def benchmark_recommendation_engine(self):
        """Benchmark integrated recommendation engine performance."""
        print("\\n" + "="*60)
        print("RECOMMENDATION ENGINE INTEGRATION BENCHMARK")
        print("="*60)
        
        # Test different recommendation strategies
        strategies = [
            "collaborative",
            "content_based", 
            "category_based",
            "hybrid"
        ]
        
        engine = RecommendationEngine()
        
        # Populate with test data
        print("\\nPopulating recommendation engine...")
        num_users, num_items = 1000, 2000
        
        start_time = time.time()
        for user_i in range(num_users):
            user_id = f"user_{user_i:05d}"
            # Each user interacts with 10-20 random items
            for _ in range(random.randint(10, 20)):
                item_id = f"item_{random.randint(0, num_items-1):05d}"
                rating = random.uniform(1.0, 5.0)
                action = random.choice(["view", "purchase", "cart_add", "rating"])
                engine.add_user_interaction(user_id, item_id, rating, action)
        
        population_time = time.time() - start_time
        print(f"Population time: {population_time:.4f}s")
        
        # Test each recommendation strategy
        strategy_results = {}
        sample_users = [f"user_{i:05d}" for i in range(0, num_users, 100)]  # Sample every 100th user
        
        for strategy in strategies:
            print(f"\\nTesting {strategy} recommendations...")
            
            start_time = time.time()
            recommendation_count = 0
            
            for user_id in sample_users:
                if strategy == "collaborative":
                    recs = engine.get_collaborative_recommendations(user_id, 10)
                elif strategy == "content_based":
                    recs = engine.get_content_based_recommendations(user_id, 10)
                elif strategy == "category_based":
                    recs = engine.get_category_based_recommendations(user_id, 10)
                else:  # hybrid
                    recs = engine.get_hybrid_recommendations(user_id, 10)
                
                recommendation_count += len(recs)
            
            strategy_time = time.time() - start_time
            avg_time_per_user = strategy_time / len(sample_users)
            
            print(f"  Total time: {strategy_time:.4f}s")
            print(f"  Avg time per user: {avg_time_per_user:.6f}s")
            print(f"  Total recommendations: {recommendation_count}")
            
            strategy_results[strategy] = {
                'total_time': strategy_time,
                'avg_time_per_user': avg_time_per_user,
                'recommendation_count': recommendation_count,
                'users_tested': len(sample_users)
            }
        
        self.results['recommendation_strategies'] = strategy_results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        report_path = os.path.join(self.output_dir, "performance_report.json")
        
        # Add system information
        self.results['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results to JSON
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate text summary
        summary_path = os.path.join(self.output_dir, "performance_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("PHASE 3 PERFORMANCE BENCHMARK SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")
            
            # Hash table optimization summary
            if 'hash_table_optimization' in self.results:
                f.write("Hash Table Optimization Results:\\n")
                f.write("-" * 30 + "\\n")
                original = self.results['hash_table_optimization']['original'][-1][1]
                optimized = self.results['hash_table_optimization']['optimized'][-1][1]
                
                insert_improvement = original['insert_time'] / optimized['insert_time']
                lookup_improvement = original['lookup_time'] / optimized['lookup_time']
                memory_savings = original['memory_mb'] - optimized['memory_mb']
                
                f.write(f"Insertion speedup: {insert_improvement:.2f}x\\n")
                f.write(f"Lookup speedup: {lookup_improvement:.2f}x\\n")
                f.write(f"Memory savings: {memory_savings:.2f} MB\\n\\n")
            
            # Scalability summary
            if 'scalability_users' in self.results:
                f.write("Scalability Analysis:\\n")
                f.write("-" * 20 + "\\n")
                user_results = self.results['scalability_users']
                f.write(f"Tested up to {user_results[-1][0]:,} users\\n")
                f.write(f"Processing time: {user_results[-1][1]['total_time']:.2f}s\\n")
                f.write(f"Memory usage: {user_results[-1][1]['memory_mb']:.2f} MB\\n\\n")
        
        print(f"\\nPerformance report saved to: {report_path}")
        print(f"Performance summary saved to: {summary_path}")
    
    def generate_visualizations(self):
        """Generate performance visualization charts."""
        try:
            # Hash table performance comparison
            if 'hash_table_optimization' in self.results:
                self._plot_hash_table_comparison()
            
            # Scalability charts
            if 'scalability_users' in self.results:
                self._plot_scalability_analysis()
            
            # Memory usage charts
            if 'memory_efficiency' in self.results:
                self._plot_memory_usage()
            
            print("Performance visualizations saved to:", self.output_dir)
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
    
    def _plot_hash_table_comparison(self):
        """Plot hash table performance comparison."""
        original = self.results['hash_table_optimization']['original']
        optimized = self.results['hash_table_optimization']['optimized']
        
        sizes = [item[0] for item in original]
        orig_insert = [item[1]['insert_time'] for item in original]
        opt_insert = [item[1]['insert_time'] for item in optimized]
        orig_lookup = [item[1]['lookup_time'] for item in original]
        opt_lookup = [item[1]['lookup_time'] for item in optimized]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Insertion time comparison
        ax1.plot(sizes, orig_insert, 'ro-', label='Original', linewidth=2)
        ax1.plot(sizes, opt_insert, 'bo-', label='Optimized', linewidth=2)
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Insertion Time (s)')
        ax1.set_title('Hash Table Insertion Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Lookup time comparison
        ax2.plot(sizes, orig_lookup, 'ro-', label='Original', linewidth=2)
        ax2.plot(sizes, opt_lookup, 'bo-', label='Optimized', linewidth=2)
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel('Lookup Time (s)')
        ax2.set_title('Hash Table Lookup Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hash_table_performance.png'), dpi=300)
        plt.close()
    
    def _plot_scalability_analysis(self):
        """Plot scalability analysis results."""
        user_results = self.results['scalability_users']
        
        users = [item[0] for item in user_results]
        times = [item[1]['total_time'] for item in user_results]
        memory = [item[1]['memory_mb'] for item in user_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Processing time scalability
        ax1.plot(users, times, 'go-', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Users')
        ax1.set_ylabel('Processing Time (s)')
        ax1.set_title('Processing Time Scalability')
        ax1.grid(True, alpha=0.3)
        
        # Memory usage scalability
        ax2.plot(users, memory, 'mo-', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Users')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Scalability')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'scalability_analysis.png'), dpi=300)
        plt.close()
    
    def _plot_memory_usage(self):
        """Plot memory usage analysis."""
        memory_data = self.results['memory_efficiency']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for structure, data in memory_data.items():
            sizes = [item[0] for item in data]
            memory = [item[1] for item in data]
            label = structure.replace('_', ' ').title()
            ax.plot(sizes, memory, 'o-', linewidth=2, markersize=6, label=label)
        
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage by Data Structure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'memory_usage.png'), dpi=300)
        plt.close()


def run_phase3_benchmarks():
    """Run complete Phase 3 performance benchmarks."""
    benchmark = PerformanceBenchmark()
    benchmark.run_comprehensive_benchmark()
    return benchmark.results


if __name__ == "__main__":
    print("Starting Phase 3 Performance Benchmarks...")
    results = run_phase3_benchmarks()
    print("\\nBenchmarking complete!")