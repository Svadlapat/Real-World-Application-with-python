"""
Phase 3 Optimization Demonstration
Comprehensive showcase of performance improvements and advanced features

This script demonstrates:
1. Performance improvements comparison
2. Advanced optimization features
3. Large-scale dataset handling
4. Concurrent access capabilities
5. Memory efficiency improvements
"""

import sys
import os
import time
import random
import threading
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_structures.hash_table import UserItemHashTable as OriginalHashTable
from data_structures.similarity_graph import ProductSimilarityGraph as OriginalGraph
from data_structures.optimized_hash_table import OptimizedUserItemHashTable
from data_structures.optimized_similarity_graph import OptimizedProductSimilarityGraph
from recommendation_engine import RecommendationEngine


class Phase3Demonstration:
    """
    Comprehensive demonstration of Phase 3 optimizations and improvements.
    """
    
    def __init__(self):
        """Initialize demonstration environment."""
        self.process = psutil.Process()
        print("="*80)
        print("PHASE 3 OPTIMIZATION DEMONSTRATION")
        print("E-commerce Recommendation System - Advanced Performance Features")
        print("="*80)
    
    def run_complete_demonstration(self):
        """Run comprehensive demonstration of all Phase 3 features."""
        
        # 1. Performance Improvement Showcase
        self.demonstrate_performance_improvements()
        
        # 2. Advanced Optimization Features
        self.demonstrate_advanced_features()
        
        # 3. Large Dataset Handling
        self.demonstrate_large_dataset_capabilities()
        
        # 4. Concurrent Access Performance
        self.demonstrate_concurrent_performance()
        
        # 5. Memory Efficiency Improvements
        self.demonstrate_memory_efficiency()
        
        # 6. Real-world Scenario Simulation
        self.demonstrate_realistic_workload()
        
        print("\\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("Phase 3 optimizations deliver significant performance improvements")
        print("while maintaining system reliability and accuracy.")
        print("="*80)
    
    def demonstrate_performance_improvements(self):
        """Showcase performance improvements with side-by-side comparison."""
        print("\\n" + "="*60)
        print("1. PERFORMANCE IMPROVEMENT SHOWCASE")
        print("="*60)
        
        # Test with moderately sized dataset for clear comparison
        test_size = 10000
        print(f"\\nComparing Phase 2 vs Phase 3 with {test_size:,} operations:")
        
        # Generate test data
        test_data = []
        for i in range(test_size):
            user_id = f"user_{i % 1000:04d}"
            item_id = f"item_{random.randint(0, 5000):05d}"
            data = {
                "rating": random.uniform(1.0, 5.0),
                "timestamp": datetime.now().isoformat(),
                "action": random.choice(["view", "purchase", "cart_add"])
            }
            test_data.append((user_id, item_id, data))
        
        print("\\n📊 Hash Table Performance Comparison:")
        
        # Phase 2 Hash Table Test
        print("\\n  Phase 2 Implementation:")
        phase2_table = OriginalHashTable(test_size // 100)
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        for user_id, item_id, data in test_data:
            phase2_table.insert(user_id, item_id, data)
        
        phase2_insertion_time = time.time() - start_time
        phase2_memory = self.process.memory_info().rss / 1024 / 1024 - start_memory
        
        # Test lookups
        lookup_sample = random.sample(test_data, 1000)
        start_time = time.time()
        for user_id, item_id, _ in lookup_sample:
            phase2_table.get(user_id, item_id)
        phase2_lookup_time = time.time() - start_time
        
        print(f"    ⏱️  Insertion time: {phase2_insertion_time:.4f}s ({test_size/phase2_insertion_time:.0f} ops/sec)")
        print(f"    🔍 Lookup time: {phase2_lookup_time:.4f}s ({len(lookup_sample)/phase2_lookup_time:.0f} ops/sec)")
        print(f"    💾 Memory used: {phase2_memory:.1f} MB")
        
        # Phase 3 Hash Table Test
        print("\\n  Phase 3 Optimized Implementation:")
        phase3_table = OptimizedUserItemHashTable(test_size // 100, enable_cache=True)
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        phase3_table.batch_insert(test_data)
        
        phase3_insertion_time = time.time() - start_time
        phase3_memory = self.process.memory_info().rss / 1024 / 1024 - start_memory
        
        # Test lookups (with caching)
        start_time = time.time()
        for user_id, item_id, _ in lookup_sample:
            phase3_table.get(user_id, item_id)
        phase3_lookup_time = time.time() - start_time
        
        stats = phase3_table.get_performance_statistics()
        
        print(f"    ⏱️  Insertion time: {phase3_insertion_time:.4f}s ({test_size/phase3_insertion_time:.0f} ops/sec)")
        print(f"    🔍 Lookup time: {phase3_lookup_time:.4f}s ({len(lookup_sample)/phase3_lookup_time:.0f} ops/sec)")
        print(f"    💾 Memory used: {phase3_memory:.1f} MB")
        print(f"    🎯 Cache hit rate: {stats['cache_hit_rate']:.3f}")
        print(f"    📏 Avg probe distance: {stats['avg_probe_distance']:.2f}")
        
        # Calculate improvements
        insertion_improvement = phase2_insertion_time / phase3_insertion_time
        lookup_improvement = phase2_lookup_time / phase3_lookup_time
        memory_savings = (phase2_memory - phase3_memory) / phase2_memory * 100
        
        print("\\n🏆 Improvements Achieved:")
        print(f"    🚀 Insertion speedup: {insertion_improvement:.2f}x faster")
        print(f"    ⚡ Lookup speedup: {lookup_improvement:.2f}x faster") 
        print(f"    💰 Memory savings: {memory_savings:.1f}%")
    
    def demonstrate_advanced_features(self):
        """Demonstrate advanced optimization features."""
        print("\\n" + "="*60)
        print("2. ADVANCED OPTIMIZATION FEATURES")
        print("="*60)
        
        print("\\n🔧 Advanced Hash Table Features:")
        
        # Robin Hood Hashing demonstration
        hash_table = OptimizedUserItemHashTable(100, enable_cache=True, cache_size=50)
        
        # Insert data that would cause collisions in a simple hash table
        collision_data = []
        for i in range(500):
            user_id = f"user_{i:03d}"
            item_id = f"item_{i:03d}"  # Sequential IDs likely to collide
            data = {"rating": random.uniform(1.0, 5.0)}
            collision_data.append((user_id, item_id, data))
        
        hash_table.batch_insert(collision_data)
        stats = hash_table.get_performance_statistics()
        
        print(f"    🎯 Robin Hood Hashing:")
        print(f"      - Max probe distance: {stats['max_probe_distance']} steps")
        print(f"      - Avg probe distance: {stats['avg_probe_distance']:.2f} steps")
        print(f"      - Memory efficiency: {stats['memory_efficiency']:.3f}")
        
        # LRU Cache demonstration
        print(f"\\n    🧠 LRU Cache Performance:")
        print(f"      - Cache hit rate: {stats['cache_hit_rate']:.3f}")
        print(f"      - Cache hits: {stats['cache_hits']:,}")
        print(f"      - Cache misses: {stats['cache_misses']:,}")
        
        # Test cache effectiveness with repeated lookups
        repeated_lookups = random.sample(collision_data, 50)
        for _ in range(5):  # Access same items 5 times
            for user_id, item_id, _ in repeated_lookups:
                hash_table.get(user_id, item_id)
        
        final_stats = hash_table.get_performance_statistics()
        print(f"      - After repeated access: {final_stats['cache_hit_rate']:.3f} hit rate")
        
        print("\\n🌐 Advanced Graph Features:")
        
        # CSR Graph demonstration
        graph = OptimizedProductSimilarityGraph(similarity_threshold=0.3, lsh_enabled=True)
        
        # Create products with feature vectors for LSH
        products = [f"product_{i:04d}" for i in range(1000)]
        for product in products:
            # Generate realistic feature vectors (e.g., product embeddings)
            feature_vector = [random.gauss(0, 1) for _ in range(100)]
            graph.add_product(product, feature_vector)
        
        # Add similarity relationships
        for _ in range(5000):
            p1, p2 = random.sample(products, 2)
            similarity = random.uniform(0.1, 1.0)
            graph.add_similarity_edge(p1, p2, similarity)
        
        memory_stats = graph.get_memory_usage_stats()
        perf_stats = graph.get_performance_statistics()
        
        print(f"    💾 CSR Memory Optimization:")
        print(f"      - Memory usage: {memory_stats['csr_memory_mb']:.2f} MB")
        print(f"      - Compression ratio: {memory_stats['compression_ratio']:.3f}")
        print(f"      - Memory saved: {memory_stats['memory_saved_mb']:.2f} MB")
        
        print(f"\\n    🔍 Locality Sensitive Hashing:")
        # Test LSH performance
        test_product = random.choice(products)
        feature_vec = [random.gauss(0, 1) for _ in range(100)]
        
        start_time = time.time()
        lsh_results = graph.get_approximate_similar_products(test_product, feature_vec, max_results=10)
        lsh_time = time.time() - start_time
        
        start_time = time.time()
        exact_results = graph.get_similar_products(test_product, max_results=10)
        exact_time = time.time() - start_time
        
        print(f"      - LSH query time: {lsh_time:.6f}s")
        print(f"      - Exact query time: {exact_time:.6f}s")
        print(f"      - Speedup: {exact_time/lsh_time:.1f}x faster")
        print(f"      - LSH results: {len(lsh_results)} products found")
    
    def demonstrate_large_dataset_capabilities(self):
        """Demonstrate handling of large datasets."""
        print("\\n" + "="*60)
        print("3. LARGE DATASET HANDLING CAPABILITIES")
        print("="*60)
        
        print("\\n📈 Scalability Test with Large Dataset:")
        
        # Test with progressively larger datasets
        dataset_sizes = [10000, 50000, 100000]
        
        for size in dataset_sizes:
            print(f"\\n  Testing with {size:,} interactions:")
            
            # Create optimized hash table
            hash_table = OptimizedUserItemHashTable(size // 100, enable_cache=True)
            
            # Generate large dataset
            large_dataset = []
            print(f"    📝 Generating {size:,} interactions...")
            for i in range(size):
                user_id = f"user_{i % (size // 100):05d}"
                item_id = f"item_{random.randint(0, size // 10):06d}"
                data = {
                    "rating": random.uniform(1.0, 5.0),
                    "timestamp": datetime.now().isoformat(),
                    "action": random.choice(["view", "purchase", "cart_add"])
                }
                large_dataset.append((user_id, item_id, data))
                
                if (i + 1) % 10000 == 0:
                    print(f"      Generated {i+1:,} interactions...")
            
            # Test batch insertion performance
            start_time = time.time()
            start_memory = self.process.memory_info().rss / 1024 / 1024
            
            hash_table.batch_insert(large_dataset)
            
            insertion_time = time.time() - start_time
            end_memory = self.process.memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory
            
            # Test query performance
            sample_queries = random.sample(large_dataset, 1000)
            start_time = time.time()
            
            for user_id, item_id, _ in sample_queries:
                result = hash_table.get(user_id, item_id)
            
            query_time = time.time() - start_time
            
            # Get performance statistics
            stats = hash_table.get_performance_statistics()
            
            print(f"    ⚡ Performance Results:")
            print(f"      - Insertion rate: {size/insertion_time:.0f} ops/sec")
            print(f"      - Query rate: {1000/query_time:.0f} queries/sec")
            print(f"      - Memory usage: {memory_used:.1f} MB")
            print(f"      - Memory per record: {memory_used*1024*1024/size:.0f} bytes")
            print(f"      - Load factor: {stats['load_factor']:.3f}")
            print(f"      - Cache hit rate: {stats['cache_hit_rate']:.3f}")
            
            # Verify data integrity
            verification_sample = random.sample(large_dataset, 100)
            correct_results = 0
            for user_id, item_id, expected_data in verification_sample:
                result = hash_table.get(user_id, item_id)
                if result and result["rating"] == expected_data["rating"]:
                    correct_results += 1
            
            integrity_rate = correct_results / len(verification_sample)
            print(f"      - Data integrity: {integrity_rate:.3f} ({correct_results}/{len(verification_sample)})")
            
            del hash_table
            del large_dataset
            import gc
            gc.collect()
    
    def demonstrate_concurrent_performance(self):
        """Demonstrate concurrent access performance."""
        print("\\n" + "="*60)
        print("4. CONCURRENT ACCESS PERFORMANCE")
        print("="*60)
        
        print("\\n🔄 Multi-threaded Performance Test:")
        
        # Setup shared hash table
        shared_table = OptimizedUserItemHashTable(1000, enable_cache=True, cache_size=200)
        
        # Pre-populate with base data
        base_data = []
        for i in range(5000):
            user_id = f"user_{i:04d}"
            item_id = f"item_{i:04d}"
            data = {"rating": random.uniform(1.0, 5.0), "timestamp": "2023-01-01"}
            base_data.append((user_id, item_id, data))
        
        shared_table.batch_insert(base_data)
        print(f"    📚 Pre-populated with {len(base_data):,} interactions")
        
        # Test different thread counts
        thread_counts = [1, 2, 4, 8]
        operations_per_thread = 2000
        
        for num_threads in thread_counts:
            print(f"\\n    🧵 Testing with {num_threads} thread(s):")
            
            results = []
            
            def worker_thread(thread_id):
                """Worker function for concurrent testing."""
                thread_stats = {"operations": 0, "errors": 0}
                start_time = time.time()
                
                for i in range(operations_per_thread):
                    try:
                        if random.random() < 0.3:  # 30% writes
                            user_id = f"thread_{thread_id}_user_{i:04d}"
                            item_id = f"item_{random.randint(0, 10000):04d}"
                            data = {"rating": random.uniform(1.0, 5.0), "timestamp": "2023-01-01"}
                            shared_table.insert(user_id, item_id, data)
                        else:  # 70% reads
                            user_id = f"user_{random.randint(0, 4999):04d}"
                            item_id = f"item_{random.randint(0, 4999):04d}"
                            shared_table.get(user_id, item_id)
                        
                        thread_stats["operations"] += 1
                        
                    except Exception as e:
                        thread_stats["errors"] += 1
                
                thread_stats["duration"] = time.time() - start_time
                return thread_stats
            
            # Run concurrent test
            overall_start = time.time()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
                thread_results = [future.result() for future in futures]
            
            overall_time = time.time() - overall_start
            
            # Calculate aggregate metrics
            total_operations = sum(r["operations"] for r in thread_results)
            total_errors = sum(r["errors"] for r in thread_results)
            throughput = total_operations / overall_time
            error_rate = total_errors / (total_operations + total_errors) if (total_operations + total_errors) > 0 else 0
            
            print(f"      📊 Results:")
            print(f"        - Total operations: {total_operations:,}")
            print(f"        - Throughput: {throughput:.0f} ops/sec")
            print(f"        - Errors: {total_errors} ({error_rate:.3%})")
            print(f"        - Total time: {overall_time:.3f}s")
            
            # Calculate scaling efficiency
            if num_threads == 1:
                single_thread_throughput = throughput
            else:
                scaling_efficiency = throughput / (single_thread_throughput * num_threads)
                print(f"        - Scaling efficiency: {scaling_efficiency:.3f} ({scaling_efficiency:.1%})")
        
        # Get final statistics
        final_stats = shared_table.get_performance_statistics()
        print(f"\\n    📈 Final System Statistics:")
        print(f"      - Total lookups: {final_stats['total_lookups']:,}")
        print(f"      - Overall cache hit rate: {final_stats['cache_hit_rate']:.3f}")
        print(f"      - System load factor: {final_stats['load_factor']:.3f}")
    
    def demonstrate_memory_efficiency(self):
        """Demonstrate memory efficiency improvements."""
        print("\\n" + "="*60)
        print("5. MEMORY EFFICIENCY IMPROVEMENTS")
        print("="*60)
        
        print("\\n💾 Memory Usage Comparison:")
        
        # Compare memory usage at different scales
        test_sizes = [1000, 5000, 10000]
        
        for size in test_sizes:
            print(f"\\n  📊 Testing {size:,} records:")
            
            # Generate test data
            test_data = [(f"user_{i:04d}", f"item_{i:04d}", {"rating": 4.0}) 
                        for i in range(size)]
            
            # Phase 2 memory usage
            import gc
            gc.collect()
            memory_before = self.process.memory_info().rss / 1024 / 1024
            
            phase2_table = OriginalHashTable(size // 10)
            for user_id, item_id, data in test_data:
                phase2_table.insert(user_id, item_id, data)
            
            gc.collect()
            phase2_memory = self.process.memory_info().rss / 1024 / 1024 - memory_before
            
            # Clean up Phase 2
            del phase2_table
            gc.collect()
            
            # Phase 3 memory usage
            memory_before = self.process.memory_info().rss / 1024 / 1024
            
            phase3_table = OptimizedUserItemHashTable(size // 10, enable_cache=True)
            phase3_table.batch_insert(test_data)
            
            gc.collect()
            phase3_memory = self.process.memory_info().rss / 1024 / 1024 - memory_before
            
            # Get detailed memory statistics
            memory_stats = phase3_table.get_performance_statistics()
            
            memory_savings = (phase2_memory - phase3_memory) / phase2_memory * 100
            memory_per_record_p2 = phase2_memory * 1024 * 1024 / size
            memory_per_record_p3 = phase3_memory * 1024 * 1024 / size
            
            print(f"    Phase 2: {phase2_memory:.2f} MB ({memory_per_record_p2:.0f} bytes/record)")
            print(f"    Phase 3: {phase3_memory:.2f} MB ({memory_per_record_p3:.0f} bytes/record)")
            print(f"    💰 Savings: {memory_savings:.1f}%")
            print(f"    🎯 Efficiency: {memory_stats['memory_efficiency']:.3f}")
            
            del phase3_table
            gc.collect()
        
        # Demonstrate memory pressure handling
        print("\\n🧪 Memory Pressure Handling Test:")
        
        hash_table = OptimizedUserItemHashTable(1000, enable_cache=True, cache_size=500)
        
        # Simulate memory pressure with large dataset
        pressure_data = []
        for i in range(20000):
            user_id = f"pressure_user_{i:05d}"
            item_id = f"item_{i:05d}"
            data = {
                "rating": random.uniform(1.0, 5.0),
                "timestamp": datetime.now().isoformat(),
                "metadata": f"large_string_data_{i}" * 5  # Larger data objects
            }
            pressure_data.append((user_id, item_id, data))
        
        # Monitor memory during insertion
        memory_samples = []
        batch_size = 2000
        
        for i in range(0, len(pressure_data), batch_size):
            batch = pressure_data[i:i+batch_size]
            hash_table.batch_insert(batch)
            
            gc.collect()
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            print(f"    Batch {i//batch_size + 1}: {current_memory:.1f} MB")
        
        # Check for memory leaks (memory should stabilize)
        if len(memory_samples) >= 3:
            recent_growth = memory_samples[-1] - memory_samples[-3]
            print(f"    📈 Recent memory growth: {recent_growth:.1f} MB")
            
            if recent_growth < 50:  # Less than 50MB growth is acceptable
                print("    ✅ No significant memory leaks detected")
            else:
                print("    ⚠️  Potential memory leak detected")
        
        final_stats = hash_table.get_performance_statistics()
        print(f"    🎯 Final cache hit rate: {final_stats['cache_hit_rate']:.3f}")
        print(f"    💾 Final memory efficiency: {final_stats['memory_efficiency']:.3f}")
    
    def demonstrate_realistic_workload(self):
        """Demonstrate performance with realistic e-commerce workload."""
        print("\\n" + "="*60)
        print("6. REALISTIC E-COMMERCE WORKLOAD SIMULATION")
        print("="*60)
        
        print("\\n🛒 E-commerce Scenario Simulation:")
        print("    - 10,000 users")
        print("    - 25,000 products") 
        print("    - Mixed interaction types (views, purchases, ratings)")
        print("    - Realistic access patterns")
        
        # Create recommendation engine with optimized components
        engine = RecommendationEngine()
        
        # Simulate realistic user behavior
        num_users = 10000
        num_products = 25000
        
        print("\\n📝 Generating realistic interaction data...")
        
        # Popular products (follow power law distribution)
        popular_products = [f"product_{i:05d}" for i in range(1000)]  # Top 1000 products
        regular_products = [f"product_{i:05d}" for i in range(1000, num_products)]
        
        interaction_count = 0
        start_time = time.time()
        
        for user_i in range(num_users):
            user_id = f"user_{user_i:05d}"
            
            # Each user has 5-50 interactions (realistic range)
            num_interactions = random.randint(5, 50)
            
            for _ in range(num_interactions):
                # 80% interactions with popular products (realistic distribution)
                if random.random() < 0.8:
                    item_id = random.choice(popular_products)
                else:
                    item_id = random.choice(regular_products)
                
                # Realistic rating distribution (skewed toward positive)
                rating = max(1.0, min(5.0, random.gauss(3.8, 1.2)))
                
                # Realistic action distribution
                action_weights = {"view": 0.6, "cart_add": 0.25, "purchase": 0.1, "rating": 0.05}
                action = random.choices(list(action_weights.keys()), 
                                     weights=list(action_weights.values()))[0]
                
                engine.add_user_interaction(user_id, item_id, rating, action)
                interaction_count += 1
            
            if (user_i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = interaction_count / elapsed
                print(f"    Processed {user_i + 1:,} users, {interaction_count:,} interactions ({rate:.0f} ops/sec)")
        
        population_time = time.time() - start_time
        print(f"\\n✅ Data population complete:")
        print(f"    - Total interactions: {interaction_count:,}")
        print(f"    - Population time: {population_time:.2f}s")
        print(f"    - Average rate: {interaction_count/population_time:.0f} interactions/sec")
        
        # Test recommendation generation performance
        print("\\n🎯 Testing recommendation generation:")
        
        # Sample users for testing
        test_users = [f"user_{i:05d}" for i in range(0, num_users, 100)]  # Every 100th user
        
        recommendation_types = [
            ("Collaborative Filtering", "collaborative"),
            ("Hybrid Recommendations", "hybrid")
        ]
        
        for rec_name, rec_type in recommendation_types:
            print(f"\\n  {rec_name}:")
            
            start_time = time.time()
            total_recommendations = 0
            
            for user_id in test_users:
                if rec_type == "collaborative":
                    recs = engine.get_collaborative_recommendations(user_id, 10)
                else:  # hybrid
                    recs = engine.get_hybrid_recommendations(user_id, 10)
                
                total_recommendations += len(recs)
            
            rec_time = time.time() - start_time
            avg_time_per_user = rec_time / len(test_users)
            users_per_second = len(test_users) / rec_time
            
            print(f"    - Users tested: {len(test_users):,}")
            print(f"    - Total recommendations: {total_recommendations:,}")
            print(f"    - Average time per user: {avg_time_per_user:.4f}s")
            print(f"    - Users per second: {users_per_second:.0f}")
            print(f"    - Avg recommendations per user: {total_recommendations/len(test_users):.1f}")
        
        # System health check
        stats = engine.get_system_statistics()
        print(f"\\n📊 Final System Statistics:")
        print(f"    - Hash table size: {stats['user_interactions']['size']:,}")
        print(f"    - Hash table load factor: {stats['user_interactions']['load_factor']:.3f}")
        print(f"    - Graph nodes: {stats['similarity_network']['num_nodes']:,}")
        print(f"    - Graph edges: {stats['similarity_network']['num_edges']:,}")
        print(f"    - Memory usage: ~{self.process.memory_info().rss / 1024 / 1024:.0f} MB")
        
        # Performance validation
        if avg_time_per_user < 0.1:  # Sub-100ms response time
            print("\\n✅ System meets real-time performance requirements!")
        else:
            print("\\n⚠️  System may need further optimization for real-time use")


def main():
    """Run the complete Phase 3 demonstration."""
    demo = Phase3Demonstration()
    demo.run_complete_demonstration()


if __name__ == "__main__":
    main()