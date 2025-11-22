"""
Phase 3 Stress Testing and Large Dataset Validation
Comprehensive testing for scalability, robustness, and performance under extreme conditions

This module provides:
1. Large dataset stress testing (up to 1M+ records)
2. Edge case and boundary condition testing
3. Memory pressure testing
4. Concurrent load testing
5. System stability validation
6. Performance regression testing
"""

import unittest
import sys
import os
import time
import random
import threading
import multiprocessing
import gc
import psutil
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from memory_profiler import profile
import weakref

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_structures.optimized_hash_table import OptimizedUserItemHashTable
from data_structures.optimized_similarity_graph import OptimizedProductSimilarityGraph
from data_structures.hash_table import UserItemHashTable
from data_structures.similarity_graph import ProductSimilarityGraph
from recommendation_engine import RecommendationEngine


class StressTestUserItemHashTable(unittest.TestCase):
    """Stress testing for optimized hash table implementation."""
    
    def setUp(self):
        """Set up stress test environment."""
        self.process = psutil.Process()
        gc.collect()
        self.initial_memory = self.process.memory_info().rss
    
    def tearDown(self):
        """Clean up after stress tests."""
        gc.collect()
    
    def test_large_dataset_insertion_performance(self):
        """Test insertion performance with large datasets (100K-1M records)."""
        print("\\nTesting large dataset insertion performance...")
        
        dataset_sizes = [100000, 500000, 1000000]
        
        for size in dataset_sizes:
            print(f"\\n  Testing {size:,} records:")
            
            # Test optimized hash table
            optimized_table = OptimizedUserItemHashTable(size // 100, enable_cache=True)
            
            # Generate large test dataset
            test_data = []
            for i in range(size):
                user_id = f"user_{i % 10000:05d}"  # 10K unique users
                item_id = f"item_{random.randint(0, size//10):06d}"
                data = {
                    "rating": random.uniform(1.0, 5.0),
                    "timestamp": datetime.now().isoformat(),
                    "action": random.choice(["view", "purchase", "cart_add", "rating"])
                }
                test_data.append((user_id, item_id, data))
                
                # Yield control periodically to prevent blocking
                if i % 10000 == 0:
                    print(f"    Generated {i:,} records...")
            
            # Test batch insertion performance
            start_time = time.time()
            start_memory = self.process.memory_info().rss
            
            optimized_table.batch_insert(test_data)
            
            insertion_time = time.time() - start_time
            end_memory = self.process.memory_info().rss
            memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
            
            # Verify data integrity
            sample_size = min(1000, size)
            sample_data = random.sample(test_data, sample_size)
            successful_lookups = 0
            
            lookup_start = time.time()
            for user_id, item_id, expected_data in sample_data:
                result = optimized_table.get(user_id, item_id)
                if result is not None and result["rating"] == expected_data["rating"]:
                    successful_lookups += 1
            lookup_time = time.time() - lookup_start
            
            # Performance assertions
            insertion_rate = size / insertion_time
            lookup_rate = sample_size / lookup_time
            
            print(f"    Insertion rate: {insertion_rate:.0f} records/second")
            print(f"    Lookup rate: {lookup_rate:.0f} lookups/second")
            print(f"    Memory used: {memory_used:.1f} MB")
            print(f"    Data integrity: {successful_lookups}/{sample_size} ({100*successful_lookups/sample_size:.1f}%)")
            
            # Performance requirements
            self.assertGreater(insertion_rate, 10000, 
                              f"Insertion rate {insertion_rate:.0f} below minimum 10K/sec")
            self.assertGreater(lookup_rate, 50000,
                              f"Lookup rate {lookup_rate:.0f} below minimum 50K/sec") 
            self.assertEqual(successful_lookups, sample_size, "Data integrity check failed")
            
            # Memory efficiency check (should be reasonable)
            memory_per_record = memory_used * 1024 * 1024 / size  # bytes per record
            self.assertLess(memory_per_record, 200, 
                           f"Memory per record {memory_per_record:.1f} bytes too high")
            
            # Clean up
            del optimized_table
            del test_data
            gc.collect()
    
    def test_hash_collision_resistance(self):
        """Test performance under high collision scenarios."""
        print("\\nTesting hash collision resistance...")
        
        # Create scenario likely to cause collisions
        hash_table = OptimizedUserItemHashTable(1000)  # Small table to force collisions
        
        # Insert many items with similar keys (high collision probability)
        collision_data = []
        for i in range(10000):
            user_id = f"user_{i:04d}"
            item_id = f"item_{i:04d}"  # Sequential IDs likely to collide
            data = {"rating": 4.0, "timestamp": "2023-01-01"}
            collision_data.append((user_id, item_id, data))
        
        start_time = time.time()
        hash_table.batch_insert(collision_data)
        insertion_time = time.time() - start_time
        
        # Test lookup performance under collisions
        start_time = time.time()
        for user_id, item_id, _ in collision_data[:1000]:  # Test 1000 lookups
            result = hash_table.get(user_id, item_id)
            self.assertIsNotNone(result)
        lookup_time = time.time() - start_time
        
        # Get collision statistics
        stats = hash_table.get_performance_statistics()
        
        print(f"    Insertion time with collisions: {insertion_time:.4f}s")
        print(f"    Lookup time with collisions: {lookup_time:.4f}s")
        print(f"    Max probe distance: {stats['max_probe_distance']}")
        print(f"    Average probe distance: {stats['avg_probe_distance']:.2f}")
        
        # Performance should still be reasonable even with collisions
        self.assertLess(lookup_time, 0.1, "Lookup time too slow under collisions")
        self.assertLess(stats['avg_probe_distance'], 3.0, "Average probe distance too high")
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure conditions."""
        print("\\nTesting memory pressure handling...")
        
        hash_table = OptimizedUserItemHashTable(10000, enable_cache=True, cache_size=1000)
        
        # Monitor memory usage during large operations
        initial_memory = self.process.memory_info().rss
        memory_samples = [initial_memory]
        
        # Gradually increase load while monitoring memory
        batch_size = 10000
        num_batches = 10
        
        for batch_num in range(num_batches):
            batch_data = []
            for i in range(batch_size):
                user_id = f"batch_{batch_num}_user_{i:05d}"
                item_id = f"item_{random.randint(0, 50000):05d}"
                data = {"rating": random.uniform(1.0, 5.0), "timestamp": "2023-01-01"}
                batch_data.append((user_id, item_id, data))
            
            hash_table.batch_insert(batch_data)
            
            # Force garbage collection and measure memory
            gc.collect()
            current_memory = self.process.memory_info().rss
            memory_samples.append(current_memory)
            
            memory_mb = (current_memory - initial_memory) / 1024 / 1024
            print(f"    Batch {batch_num + 1}: {memory_mb:.1f} MB used")
            
            # Check for memory leaks (memory should not grow excessively)
            if batch_num > 0:
                memory_growth = (memory_samples[-1] - memory_samples[-2]) / 1024 / 1024
                # Allow some growth but flag excessive growth
                if memory_growth > 100:  # More than 100MB per batch is concerning
                    print(f"    Warning: Large memory growth detected: {memory_growth:.1f} MB")
        
        # Test cache effectiveness under pressure
        stats = hash_table.get_performance_statistics()
        print(f"    Final cache hit rate: {stats['cache_hit_rate']:.3f}")
        print(f"    Total memory efficiency: {stats['memory_efficiency']:.3f}")
        
        # Cache should be reasonably effective
        self.assertGreater(stats['cache_hit_rate'], 0.1, "Cache hit rate too low")
        self.assertGreater(stats['memory_efficiency'], 0.5, "Memory efficiency too low")
    
    def test_concurrent_stress_load(self):
        """Test concurrent access under high load."""
        print("\\nTesting concurrent stress load...")
        
        hash_table = OptimizedUserItemHashTable(5000, enable_cache=True)
        
        # Pre-populate with base data
        for i in range(5000):
            user_id = f"user_{i:04d}"
            item_id = f"item_{i:04d}"
            data = {"rating": 4.0, "timestamp": "2023-01-01"}
            hash_table.insert(user_id, item_id, data)
        
        # Concurrent stress test configuration
        num_threads = 8
        operations_per_thread = 5000
        results = {}
        errors = []
        
        def stress_worker(thread_id):
            """Worker function for stress testing."""
            try:
                thread_results = {
                    'operations': 0,
                    'errors': 0,
                    'start_time': time.time()
                }
                
                for i in range(operations_per_thread):
                    try:
                        operation = random.choice(['read', 'write', 'read', 'read'])  # 75% reads
                        
                        if operation == 'write':
                            user_id = f"stress_user_{thread_id}_{i:05d}"
                            item_id = f"item_{random.randint(0, 10000):05d}"
                            data = {"rating": random.uniform(1.0, 5.0), "timestamp": "2023-01-01"}
                            hash_table.insert(user_id, item_id, data)
                        else:  # read
                            user_id = f"user_{random.randint(0, 4999):04d}"
                            item_id = f"item_{random.randint(0, 4999):04d}"
                            result = hash_table.get(user_id, item_id)
                        
                        thread_results['operations'] += 1
                        
                    except Exception as e:
                        thread_results['errors'] += 1
                        errors.append(f"Thread {thread_id}: {str(e)}")
                
                thread_results['end_time'] = time.time()
                thread_results['duration'] = thread_results['end_time'] - thread_results['start_time']
                return thread_results
                
            except Exception as e:
                errors.append(f"Thread {thread_id} failed: {str(e)}")
                return {'operations': 0, 'errors': 1, 'duration': 0}
        
        # Run concurrent stress test
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_threads)]
            thread_results = [future.result() for future in futures]
        total_time = time.time() - start_time
        
        # Analyze results
        total_operations = sum(r['operations'] for r in thread_results)
        total_errors = sum(r['errors'] for r in thread_results)
        ops_per_second = total_operations / total_time
        
        print(f"    Total operations: {total_operations:,}")
        print(f"    Total errors: {total_errors}")
        print(f"    Operations/second: {ops_per_second:.0f}")
        print(f"    Error rate: {100*total_errors/total_operations:.3f}%")
        
        # Print any errors encountered
        if errors:
            print("    Errors encountered:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"      {error}")
        
        # Performance requirements for concurrent access
        self.assertGreater(ops_per_second, 5000, 
                          f"Concurrent throughput {ops_per_second:.0f} too low")
        self.assertLess(total_errors / total_operations, 0.01, 
                       f"Error rate {100*total_errors/total_operations:.3f}% too high")


class StressTestSimilarityGraph(unittest.TestCase):
    """Stress testing for optimized similarity graph."""
    
    def test_large_graph_construction(self):
        """Test construction of large similarity graphs."""
        print("\\nTesting large graph construction...")
        
        graph_sizes = [(1000, 5000), (5000, 25000), (10000, 100000)]  # (products, edges)
        
        for num_products, num_edges in graph_sizes:
            print(f"\\n  Testing {num_products:,} products, {num_edges:,} edges:")
            
            graph = OptimizedProductSimilarityGraph(similarity_threshold=0.3, lsh_enabled=True)
            
            # Generate products
            products = [f"product_{i:06d}" for i in range(num_products)]
            
            # Add products with random feature vectors
            start_time = time.time()
            for product in products:
                feature_vec = np.random.randn(100)  # 100-dimensional features
                graph.add_product(product, feature_vec)
            product_time = time.time() - start_time
            
            # Generate and add similarity edges
            start_time = time.time()
            edges_added = 0
            for _ in range(num_edges):
                p1 = random.choice(products)
                p2 = random.choice(products)
                if p1 != p2:
                    similarity = random.uniform(0.0, 1.0)
                    graph.add_similarity_edge(p1, p2, similarity)
                    edges_added += 1
            
            edge_time = time.time() - start_time
            
            print(f"    Product addition: {product_time:.4f}s ({num_products/product_time:.0f} products/sec)")
            print(f"    Edge addition: {edge_time:.4f}s ({edges_added/edge_time:.0f} edges/sec)")
            
            # Test query performance
            start_time = time.time()
            query_count = min(100, num_products)
            total_results = 0
            
            for _ in range(query_count):
                test_product = random.choice(products)
                similar_products = graph.get_similar_products(test_product, max_results=10)
                total_results += len(similar_products)
            
            query_time = time.time() - start_time
            avg_query_time = query_time / query_count
            
            print(f"    Query performance: {avg_query_time:.6f}s per query")
            print(f"    Average results per query: {total_results/query_count:.1f}")
            
            # Memory usage analysis
            memory_stats = graph.get_memory_usage_stats()
            print(f"    Memory usage: {memory_stats['csr_memory_mb']:.1f} MB")
            print(f"    Compression ratio: {memory_stats['compression_ratio']:.3f}")
            
            # Performance assertions
            self.assertLess(avg_query_time, 0.001, "Query time too slow for large graph")
            self.assertLess(memory_stats['compression_ratio'], 0.8, "Memory compression insufficient")
            
            del graph
            gc.collect()
    
    def test_graph_clustering_performance(self):
        """Test graph clustering with large datasets."""
        print("\\nTesting graph clustering performance...")
        
        graph = OptimizedProductSimilarityGraph()
        
        # Create a graph with community structure
        num_communities = 5
        products_per_community = 200
        total_products = num_communities * products_per_community
        
        products = []
        community_assignments = {}
        
        # Generate products with community structure
        for community in range(num_communities):
            for i in range(products_per_community):
                product = f"community_{community}_product_{i:03d}"
                products.append(product)
                community_assignments[product] = community
                graph.add_product(product)
        
        # Add intra-community edges (high similarity)
        for community in range(num_communities):
            community_products = [p for p in products if community_assignments[p] == community]
            for _ in range(len(community_products) * 3):  # Dense within communities
                p1 = random.choice(community_products)
                p2 = random.choice(community_products)
                if p1 != p2:
                    similarity = random.uniform(0.7, 1.0)  # High similarity
                    graph.add_similarity_edge(p1, p2, similarity)
        
        # Add inter-community edges (lower similarity)
        for _ in range(total_products):  # Sparse between communities
            p1 = random.choice(products)
            p2 = random.choice(products)
            if community_assignments[p1] != community_assignments[p2]:
                similarity = random.uniform(0.1, 0.4)  # Lower similarity
                graph.add_similarity_edge(p1, p2, similarity)
        
        # Test clustering performance
        start_time = time.time()
        clusters = graph.compute_graph_clustering(num_clusters=num_communities)
        clustering_time = time.time() - start_time
        
        print(f"    Clustering time: {clustering_time:.4f}s")
        print(f"    Products clustered: {len(clusters):,}")
        
        # Analyze clustering quality (products from same community should be clustered together)
        community_purity = {}
        for community in range(num_communities):
            community_products = [p for p in products if community_assignments[p] == community]
            if community_products:
                cluster_assignments = [clusters.get(p, -1) for p in community_products]
                most_common_cluster = max(set(cluster_assignments), key=cluster_assignments.count)
                purity = cluster_assignments.count(most_common_cluster) / len(cluster_assignments)
                community_purity[community] = purity
        
        avg_purity = sum(community_purity.values()) / len(community_purity)
        print(f"    Average community purity: {avg_purity:.3f}")
        
        # Clustering should be reasonably fast and accurate
        self.assertLess(clustering_time, 5.0, "Clustering too slow")
        self.assertGreater(avg_purity, 0.5, "Clustering quality too low")


class StressTestRecommendationEngine(unittest.TestCase):
    """Stress testing for integrated recommendation engine."""
    
    def test_large_scale_recommendations(self):
        """Test recommendation generation with large user/item datasets."""
        print("\\nTesting large-scale recommendation generation...")
        
        engine = RecommendationEngine()
        
        # Generate large dataset
        num_users = 10000
        num_items = 50000
        interactions_per_user = 20
        
        print(f"    Generating dataset: {num_users:,} users, {num_items:,} items")
        
        # Populate with interactions
        start_time = time.time()
        total_interactions = 0
        
        for user_i in range(num_users):
            user_id = f"user_{user_i:06d}"
            
            # Each user interacts with random items
            user_items = random.sample(range(num_items), interactions_per_user)
            for item_i in user_items:
                item_id = f"item_{item_i:06d}"
                rating = random.uniform(1.0, 5.0)
                action = random.choice(["view", "purchase", "cart_add", "rating"])
                
                engine.add_user_interaction(user_id, item_id, rating, action)
                total_interactions += 1
            
            if (user_i + 1) % 1000 == 0:
                print(f"    Processed {user_i + 1:,} users...")
        
        population_time = time.time() - start_time
        interaction_rate = total_interactions / population_time
        
        print(f"    Population time: {population_time:.2f}s")
        print(f"    Interaction rate: {interaction_rate:.0f} interactions/sec")
        print(f"    Total interactions: {total_interactions:,}")
        
        # Test recommendation generation performance
        test_users = [f"user_{i:06d}" for i in range(0, num_users, 100)]  # Every 100th user
        
        recommendation_strategies = [
            ("Collaborative", "collaborative"),
            ("Hybrid", "hybrid")
        ]
        
        for strategy_name, strategy_type in recommendation_strategies:
            print(f"\\n    Testing {strategy_name} recommendations:")
            
            start_time = time.time()
            total_recommendations = 0
            
            for user_id in test_users:
                if strategy_type == "collaborative":
                    recs = engine.get_collaborative_recommendations(user_id, 10)
                else:  # hybrid
                    recs = engine.get_hybrid_recommendations(user_id, 10)
                
                total_recommendations += len(recs)
            
            recommendation_time = time.time() - start_time
            avg_time_per_user = recommendation_time / len(test_users)
            recs_per_second = len(test_users) / recommendation_time
            
            print(f"      Total time: {recommendation_time:.4f}s")
            print(f"      Avg time per user: {avg_time_per_user:.6f}s")
            print(f"      Users/second: {recs_per_second:.0f}")
            print(f"      Total recommendations: {total_recommendations}")
            
            # Performance requirements
            self.assertLess(avg_time_per_user, 0.1, 
                           f"{strategy_name} recommendations too slow")
            self.assertGreater(total_recommendations, 0, 
                              f"No {strategy_name} recommendations generated")
        
        # Test system statistics
        stats = engine.get_system_statistics()
        print(f"\\n    System statistics:")
        print(f"      Hash table size: {stats['user_interactions']['size']:,}")
        print(f"      Graph nodes: {stats['similarity_network']['num_nodes']:,}")
        print(f"      Graph edges: {stats['similarity_network']['num_edges']:,}")
    
    def test_concurrent_recommendation_load(self):
        """Test concurrent recommendation generation under load."""
        print("\\nTesting concurrent recommendation load...")
        
        engine = RecommendationEngine()
        
        # Pre-populate with smaller dataset for this test
        num_users = 1000
        num_items = 5000
        
        for user_i in range(num_users):
            user_id = f"user_{user_i:04d}"
            for _ in range(10):  # 10 interactions per user
                item_id = f"item_{random.randint(0, num_items-1):04d}"
                rating = random.uniform(1.0, 5.0)
                engine.add_user_interaction(user_id, item_id, rating, "rating")
        
        # Concurrent recommendation test
        num_threads = 8
        requests_per_thread = 50
        
        def recommendation_worker(thread_id):
            """Worker for concurrent recommendation requests."""
            results = {'recommendations': 0, 'errors': 0, 'time': 0}
            
            start_time = time.time()
            for i in range(requests_per_thread):
                try:
                    user_id = f"user_{random.randint(0, num_users-1):04d}"
                    recs = engine.get_hybrid_recommendations(user_id, 5)
                    results['recommendations'] += len(recs)
                except Exception as e:
                    results['errors'] += 1
            
            results['time'] = time.time() - start_time
            return results
        
        # Run concurrent test
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(recommendation_worker, i) for i in range(num_threads)]
            thread_results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Analyze results
        total_requests = num_threads * requests_per_thread
        total_recommendations = sum(r['recommendations'] for r in thread_results)
        total_errors = sum(r['errors'] for r in thread_results)
        requests_per_second = total_requests / total_time
        
        print(f"    Concurrent requests: {total_requests}")
        print(f"    Total recommendations: {total_recommendations}")
        print(f"    Errors: {total_errors}")
        print(f"    Requests/second: {requests_per_second:.0f}")
        print(f"    Error rate: {100*total_errors/total_requests:.2f}%")
        
        # Performance requirements
        self.assertGreater(requests_per_second, 50, "Concurrent throughput too low")
        self.assertLess(total_errors / total_requests, 0.05, "Too many errors under load")


def run_stress_tests():
    """Run comprehensive stress test suite."""
    print("=" * 80)
    print("PHASE 3 STRESS TESTING SUITE")
    print("Advanced scalability and robustness validation")
    print("=" * 80)
    
    # Create test suite
    test_classes = [
        StressTestUserItemHashTable,
        StressTestSimilarityGraph,
        StressTestRecommendationEngine
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class in test_classes:
        print(f"\\n{'='*60}")
        print(f"STRESS TESTING: {test_class.__name__.replace('StressTest', '')}")
        print(f"{'='*60}")
        
        # Create test suite for this class
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Run tests with detailed output
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        # Accumulate statistics
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        # Print summary for this test class
        print(f"\\nStress Test Class Summary:")
        print(f"  Tests run: {result.testsRun}")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
    
    # Print overall summary
    print(f"\\n{'='*80}")
    print("STRESS TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total stress tests: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    success_rate = (total_tests - total_failures - total_errors) / total_tests * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("\\n ALL STRESS TESTS PASSED!")
        print("System demonstrates excellent scalability and robustness.")
    else:
        print(f"\\n {total_failures + total_errors} stress tests failed.")
        print("Review failures to identify scalability issues.")
    
    return success_rate >= 90.0  # 90% success rate threshold


if __name__ == "__main__":
    print("Starting Phase 3 Stress Testing...")
    success = run_stress_tests()
    if success:
        print("\\nStress testing completed successfully!")
    else:
        print("\\nStress testing revealed issues that need attention.")
        sys.exit(1)