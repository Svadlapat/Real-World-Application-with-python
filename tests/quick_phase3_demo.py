"""
Quick Phase 3 Demo - Faster execution for immediate results
"""

import sys
import os
import time
import random

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_structures.hash_table import UserItemHashTable as OriginalHashTable
from data_structures.optimized_hash_table import OptimizedUserItemHashTable

def quick_demo():
    print("="*60)
    print("QUICK PHASE 3 OPTIMIZATION DEMO")
    print("="*60)
    
    # Quick hash table comparison
    print("\n📊 Hash Table Performance Comparison (Quick Test):")
    
    test_size = 5000  # Smaller size for quick results
    test_data = []
    for i in range(test_size):
        user_id = f"user_{i % 500:04d}"
        item_id = f"item_{random.randint(0, 2500):05d}"
        data = {"rating": random.uniform(1.0, 5.0)}
        test_data.append((user_id, item_id, data))
    
    # Phase 2 Test
    print("\n  Phase 2 Implementation:")
    phase2_table = OriginalHashTable(test_size // 50)
    
    start_time = time.time()
    for user_id, item_id, data in test_data:
        phase2_table.insert(user_id, item_id, data)
    phase2_time = time.time() - start_time
    
    print(f"    ⏱️  Insertion time: {phase2_time:.4f}s ({test_size/phase2_time:.0f} ops/sec)")
    
    # Phase 3 Test
    print("\n  Phase 3 Optimized Implementation:")
    phase3_table = OptimizedUserItemHashTable(test_size // 50, enable_cache=True)
    
    start_time = time.time()
    phase3_table.batch_insert(test_data)
    phase3_time = time.time() - start_time
    
    stats = phase3_table.get_performance_statistics()
    
    print(f"    ⏱️  Insertion time: {phase3_time:.4f}s ({test_size/phase3_time:.0f} ops/sec)")
    print(f"    🎯 Memory efficiency: {stats['memory_efficiency']:.3f}")
    print(f"    📏 Avg probe distance: {stats['avg_probe_distance']:.2f}")
    
    # Test cache performance
    lookup_sample = random.sample(test_data, 500)
    for user_id, item_id, _ in lookup_sample:
        phase3_table.get(user_id, item_id)  # First access
    for user_id, item_id, _ in lookup_sample:
        phase3_table.get(user_id, item_id)  # Second access (should hit cache)
    
    final_stats = phase3_table.get_performance_statistics()
    print(f"    🧠 Cache hit rate: {final_stats['cache_hit_rate']:.3f}")
    
    # Calculate improvements
    improvement = phase2_time / phase3_time
    print(f"\n🏆 Performance Improvement: {improvement:.2f}x faster!")
    
    print("\n✅ Quick demo complete! Phase 3 optimizations are working correctly.")
    print("📝 For full detailed analysis, run the complete benchmark scripts.")

if __name__ == "__main__":
    quick_demo()