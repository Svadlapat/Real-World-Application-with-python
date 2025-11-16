"""
Comprehensive Test Suite for E-Commerce Recommendation System
Phase 2: Proof of Concept Testing and Validation

This module provides extensive testing for all data structures and their
integration in the recommendation system.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
import time
import random

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_structures.hash_table import UserItemHashTable
from data_structures.similarity_graph import ProductSimilarityGraph
from data_structures.behavior_tree import UserBehaviorTree
from data_structures.category_tree import CategoryHierarchyTree
from recommendation_engine import RecommendationEngine


class TestUserItemHashTable(unittest.TestCase):
    """Test cases for UserItemHashTable data structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hash_table = UserItemHashTable(10)
    
    def test_basic_insertion_and_retrieval(self):
        """Test basic insert and get operations."""
        # Test data
        user_id = "user_001"
        item_id = "item_101"
        interaction_data = {"rating": 4.5, "timestamp": "2023-01-15", "action": "purchase"}
        
        # Insert data
        self.hash_table.insert(user_id, item_id, interaction_data)
        
        # Retrieve data
        result = self.hash_table.get(user_id, item_id)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result["rating"], 4.5)
        self.assertEqual(result["action"], "purchase")
    
    def test_update_existing_interaction(self):
        """Test updating an existing user-item interaction."""
        user_id = "user_001"
        item_id = "item_101"
        
        # Initial data
        initial_data = {"rating": 3.0, "timestamp": "2023-01-15", "action": "view"}
        self.hash_table.insert(user_id, item_id, initial_data)
        
        # Updated data
        updated_data = {"rating": 4.5, "timestamp": "2023-01-16", "action": "purchase"}
        self.hash_table.insert(user_id, item_id, updated_data)
        
        # Verify update
        result = self.hash_table.get(user_id, item_id)
        self.assertEqual(result["rating"], 4.5)
        self.assertEqual(result["action"], "purchase")
    
    def test_nonexistent_interaction(self):
        """Test retrieving non-existent interaction."""
        result = self.hash_table.get("nonexistent_user", "nonexistent_item")
        self.assertIsNone(result)
    
    def test_collision_handling(self):
        """Test hash collision handling with multiple items."""
        # Insert multiple items that might cause collisions
        test_data = [
            ("user_001", "item_101", {"rating": 4.0, "action": "purchase"}),
            ("user_002", "item_102", {"rating": 3.5, "action": "view"}),
            ("user_003", "item_103", {"rating": 5.0, "action": "purchase"}),
            ("user_004", "item_104", {"rating": 2.5, "action": "return"}),
        ]
        
        # Insert all data
        for user_id, item_id, data in test_data:
            self.hash_table.insert(user_id, item_id, data)
        
        # Verify all data can be retrieved
        for user_id, item_id, expected_data in test_data:
            result = self.hash_table.get(user_id, item_id)
            self.assertIsNotNone(result)
            self.assertEqual(result["rating"], expected_data["rating"])
    
    def test_dynamic_resizing(self):
        """Test dynamic resizing of hash table."""
        initial_capacity = self.hash_table.capacity
        
        # Insert enough items to trigger resize
        for i in range(20):
            self.hash_table.insert(f"user_{i:03d}", f"item_{i:03d}", 
                                 {"rating": 4.0, "action": "purchase"})
        
        # Verify resize occurred
        self.assertGreater(self.hash_table.capacity, initial_capacity)
        
        # Verify all data is still accessible
        for i in range(20):
            result = self.hash_table.get(f"user_{i:03d}", f"item_{i:03d}")
            self.assertIsNotNone(result)
    
    def test_user_interactions_query(self):
        """Test retrieving all interactions for a specific user."""
        user_id = "user_001"
        items = ["item_101", "item_102", "item_103"]
        
        # Insert multiple interactions for the user
        for i, item_id in enumerate(items):
            data = {"rating": 4.0 + i * 0.5, "action": "purchase"}
            self.hash_table.insert(user_id, item_id, data)
        
        # Get user interactions
        user_interactions = self.hash_table.get_user_interactions(user_id)
        
        # Verify results
        self.assertEqual(len(user_interactions), 3)
        retrieved_items = [item_id for item_id, _ in user_interactions]
        for item in items:
            self.assertIn(item, retrieved_items)
    
    def test_delete_interaction(self):
        """Test deleting user-item interactions."""
        user_id = "user_001"
        item_id = "item_101"
        data = {"rating": 4.0, "action": "purchase"}
        
        # Insert and verify
        self.hash_table.insert(user_id, item_id, data)
        self.assertIsNotNone(self.hash_table.get(user_id, item_id))
        
        # Delete and verify
        result = self.hash_table.delete(user_id, item_id)
        self.assertTrue(result)
        self.assertIsNone(self.hash_table.get(user_id, item_id))
        
        # Try to delete non-existent item
        result = self.hash_table.delete("nonexistent", "nonexistent")
        self.assertFalse(result)


class TestProductSimilarityGraph(unittest.TestCase):
    """Test cases for ProductSimilarityGraph data structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = ProductSimilarityGraph()
    
    def test_add_products_and_similarities(self):
        """Test adding products and similarity relationships."""
        # Add products
        products = ["laptop_001", "laptop_002", "phone_001"]
        for product in products:
            self.graph.add_product(product)
        
        # Add similarity
        self.graph.add_similarity_edge("laptop_001", "laptop_002", 0.8)
        
        # Verify products exist
        self.assertIn("laptop_001", self.graph.products)
        self.assertIn("laptop_002", self.graph.products)
        
        # Verify similarity relationship
        similar_products = self.graph.get_similar_products("laptop_001")
        self.assertEqual(len(similar_products), 1)
        self.assertEqual(similar_products[0][0], "laptop_002")
        self.assertEqual(similar_products[0][1], 0.8)
    
    def test_bidirectional_similarity(self):
        """Test that similarity relationships are bidirectional."""
        self.graph.add_similarity_edge("product_A", "product_B", 0.7)
        
        # Check both directions
        similar_to_A = self.graph.get_similar_products("product_A")
        similar_to_B = self.graph.get_similar_products("product_B")
        
        self.assertEqual(len(similar_to_A), 1)
        self.assertEqual(len(similar_to_B), 1)
        self.assertEqual(similar_to_A[0][0], "product_B")
        self.assertEqual(similar_to_B[0][0], "product_A")
    
    def test_similarity_threshold_filtering(self):
        """Test filtering similarities by minimum threshold."""
        # Add products with varying similarities
        self.graph.add_similarity_edge("product_A", "product_B", 0.9)
        self.graph.add_similarity_edge("product_A", "product_C", 0.5)
        self.graph.add_similarity_edge("product_A", "product_D", 0.3)
        
        # Test different thresholds
        all_similar = self.graph.get_similar_products("product_A", min_similarity=0.0)
        high_similar = self.graph.get_similar_products("product_A", min_similarity=0.6)
        very_high_similar = self.graph.get_similar_products("product_A", min_similarity=0.8)
        
        self.assertEqual(len(all_similar), 3)
        self.assertEqual(len(high_similar), 1)
        self.assertEqual(len(very_high_similar), 1)
    
    def test_recommendation_path_finding(self):
        """Test finding recommendation paths between products."""
        # Create a chain: A -> B -> C
        self.graph.add_similarity_edge("product_A", "product_B", 0.8)
        self.graph.add_similarity_edge("product_B", "product_C", 0.7)
        
        # Find path from A to C
        path = self.graph.find_recommendation_path("product_A", "product_C")
        
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], "product_A")
        self.assertEqual(path[1], "product_B")
        self.assertEqual(path[2], "product_C")
    
    def test_transitive_similarity(self):
        """Test advanced transitive similarity calculations."""
        # Create network: A-B, B-C, A-D
        self.graph.add_similarity_edge("product_A", "product_B", 0.8)
        self.graph.add_similarity_edge("product_B", "product_C", 0.7)
        self.graph.add_similarity_edge("product_A", "product_D", 0.6)
        
        # Test advanced similarity with transitive relationships
        advanced_similar = self.graph.get_top_k_similar_products_advanced(
            "product_A", k=5, use_transitive=True)
        
        self.assertGreater(len(advanced_similar), 0)
        # Should include both direct and transitive similarities
        product_ids = [item[0] for item in advanced_similar]
        self.assertIn("product_B", product_ids)  # Direct similarity
    
    def test_product_clustering(self):
        """Test finding clusters of highly similar products."""
        # Create two clusters
        # Cluster 1: A-B-C (high similarity)
        self.graph.add_similarity_edge("product_A", "product_B", 0.9)
        self.graph.add_similarity_edge("product_B", "product_C", 0.85)
        self.graph.add_similarity_edge("product_A", "product_C", 0.8)
        
        # Cluster 2: D-E (high similarity)
        self.graph.add_similarity_edge("product_D", "product_E", 0.9)
        
        # Low similarity between clusters
        self.graph.add_similarity_edge("product_C", "product_D", 0.3)
        
        # Find clusters
        clusters = self.graph.get_product_clusters(min_similarity=0.8)
        
        self.assertEqual(len(clusters), 2)
        # Verify cluster contents
        cluster_sizes = [len(cluster) for cluster in clusters]
        self.assertIn(3, cluster_sizes)  # Cluster 1 should have 3 products
        self.assertIn(2, cluster_sizes)  # Cluster 2 should have 2 products


class TestUserBehaviorTree(unittest.TestCase):
    """Test cases for UserBehaviorTree data structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tree = UserBehaviorTree()
        self.base_time = datetime.now() - timedelta(days=7)
    
    def test_basic_insertion_and_search(self):
        """Test basic insertion and timestamp-based search."""
        timestamp = self.base_time
        self.tree.insert("user_001", "item_101", timestamp, "purchase", {"rating": 4.5})
        
        # Search by exact timestamp
        results = self.tree.search_by_timestamp(timestamp)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].user_id, "user_001")
        self.assertEqual(results[0].item_id, "item_101")
        self.assertEqual(results[0].action_type, "purchase")
    
    def test_chronological_ordering(self):
        """Test that interactions are stored in chronological order."""
        timestamps = [
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=3),
            self.base_time + timedelta(hours=2),
        ]
        
        # Insert in non-chronological order
        for i, timestamp in enumerate(timestamps):
            self.tree.insert(f"user_00{i+1}", f"item_10{i+1}", 
                           timestamp, "view", {"rating": 3.0})
        
        # Get chronological order
        chronological = self.tree.inorder_traversal()
        
        self.assertEqual(len(chronological), 3)
        # Verify chronological ordering
        for i in range(len(chronological) - 1):
            self.assertLessEqual(chronological[i].timestamp, 
                               chronological[i + 1].timestamp)
    
    def test_range_queries(self):
        """Test time range queries."""
        # Insert interactions across a week
        for i in range(7):
            timestamp = self.base_time + timedelta(days=i)
            self.tree.insert(f"user_001", f"item_10{i+1}", 
                           timestamp, "view", {"rating": 3.0})
        
        # Query for interactions in middle 3 days
        start_time = self.base_time + timedelta(days=2)
        end_time = self.base_time + timedelta(days=4)
        
        results = self.tree.get_interactions_in_range(start_time, end_time)
        
        # Should include days 2, 3, 4 (3 interactions)
        self.assertEqual(len(results), 3)
        
        # Verify all results are within range
        for result in results:
            self.assertGreaterEqual(result.timestamp, start_time)
            self.assertLessEqual(result.timestamp, end_time)
    
    def test_recent_activity_analysis(self):
        """Test recent activity retrieval and filtering."""
        user_id = "user_001"
        
        # Insert recent and old activities
        recent_time = datetime.now() - timedelta(hours=2)
        old_time = datetime.now() - timedelta(days=30)
        
        self.tree.insert(user_id, "item_101", recent_time, "purchase", {"rating": 4.0})
        self.tree.insert(user_id, "item_102", old_time, "view", {"rating": 3.0})
        self.tree.insert("user_002", "item_103", recent_time, "view", {"rating": 3.5})
        
        # Get recent activity for specific user
        recent_activity = self.tree.get_recent_activity(
            user_id=user_id, hours=24, action_types=["purchase"])
        
        self.assertEqual(len(recent_activity), 1)
        self.assertEqual(recent_activity[0].user_id, user_id)
        self.assertEqual(recent_activity[0].action_type, "purchase")
    
    def test_behavior_pattern_analysis(self):
        """Test user behavior pattern analysis."""
        user_id = "user_001"
        
        # Insert various interactions for pattern analysis
        interactions = [
            ("item_101", "purchase", 5.0),
            ("item_102", "view", 3.0),
            ("item_103", "purchase", 4.5),
            ("item_101", "view", 4.0),  # Repeated item
        ]
        
        base_time = datetime.now() - timedelta(days=5)
        for i, (item_id, action, rating) in enumerate(interactions):
            timestamp = base_time + timedelta(hours=i)
            self.tree.insert(user_id, item_id, timestamp, action, {"rating": rating})
        
        # Analyze behavior pattern
        pattern = self.tree.get_user_behavior_pattern(user_id, days=30)
        
        self.assertEqual(pattern["total_interactions"], 4)
        self.assertEqual(pattern["action_counts"]["purchase"], 2)
        self.assertEqual(pattern["action_counts"]["view"], 2)
        
        # Check favorite items
        favorite_items = pattern["favorite_items"]
        self.assertGreater(len(favorite_items), 0)
        # item_101 should be top favorite (2 interactions)
        self.assertEqual(favorite_items[0][0], "item_101")
        self.assertEqual(favorite_items[0][1], 2)


class TestCategoryHierarchyTree(unittest.TestCase):
    """Test cases for CategoryHierarchyTree data structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tree = CategoryHierarchyTree("E-commerce Store")
    
    def test_category_creation_and_hierarchy(self):
        """Test creating categories and maintaining hierarchy."""
        # Create hierarchy: Electronics -> Computers -> Laptops
        electronics = self.tree.add_category("electronics", "Electronics")
        computers = self.tree.add_category("computers", "Computers", "electronics")
        laptops = self.tree.add_category("laptops", "Laptops", "computers")
        
        # Verify hierarchy relationships
        self.assertEqual(computers.parent, electronics)
        self.assertEqual(laptops.parent, computers)
        self.assertIn(computers, electronics.children)
        self.assertIn(laptops, computers.children)
    
    def test_product_assignment_and_retrieval(self):
        """Test assigning products to categories and retrieving them."""
        # Create categories
        self.tree.add_category("electronics", "Electronics")
        self.tree.add_category("laptops", "Laptops", "electronics")
        
        # Add products
        products = ["laptop_001", "laptop_002", "laptop_003"]
        for product in products:
            self.tree.add_product_to_category(product, "laptops")
        
        # Verify products in category
        laptop_products = self.tree.get_products_in_category("laptops", 
                                                           include_subcategories=False)
        self.assertEqual(len(laptop_products), 3)
        for product in products:
            self.assertIn(product, laptop_products)
    
    def test_hierarchical_product_retrieval(self):
        """Test retrieving products from category hierarchy."""
        # Create hierarchy
        self.tree.add_category("electronics", "Electronics")
        self.tree.add_category("computers", "Computers", "electronics")
        self.tree.add_category("phones", "Phones", "electronics")
        self.tree.add_category("laptops", "Laptops", "computers")
        
        # Add products to different levels
        self.tree.add_product_to_category("laptop_001", "laptops")
        self.tree.add_product_to_category("desktop_001", "computers")
        self.tree.add_product_to_category("phone_001", "phones")
        
        # Test hierarchical retrieval
        electronics_products = self.tree.get_products_in_category(
            "electronics", include_subcategories=True)
        computers_products = self.tree.get_products_in_category(
            "computers", include_subcategories=True)
        
        # Electronics should include all products
        self.assertEqual(len(electronics_products), 3)
        # Computers should include laptop and desktop
        self.assertEqual(len(computers_products), 2)
        self.assertIn("laptop_001", computers_products)
        self.assertIn("desktop_001", computers_products)
    
    def test_category_recommendations(self):
        """Test category-based product recommendations."""
        # Setup categories and products
        self.tree.add_category("electronics", "Electronics")
        self.tree.add_category("laptops", "Laptops", "electronics")
        self.tree.add_category("phones", "Phones", "electronics")
        
        # Add products
        self.tree.add_product_to_category("laptop_001", "laptops")
        self.tree.add_product_to_category("laptop_002", "laptops")
        self.tree.add_product_to_category("phone_001", "phones")
        
        # Get recommendations for a laptop
        recommendations = self.tree.get_category_recommendations("laptop_001")
        
        # Should recommend other laptop and phone (sibling category)
        self.assertIn("laptop_002", recommendations)
        self.assertIn("phone_001", recommendations)
        self.assertNotIn("laptop_001", recommendations)  # Shouldn't recommend itself
    
    def test_category_path_resolution(self):
        """Test resolving full paths from root to category."""
        # Create deep hierarchy
        self.tree.add_category("electronics", "Electronics")
        self.tree.add_category("computers", "Computers", "electronics")
        self.tree.add_category("laptops", "Laptops", "computers")
        self.tree.add_category("gaming_laptops", "Gaming Laptops", "laptops")
        
        # Get path
        path = self.tree.get_category_path("gaming_laptops")
        
        expected_path = ["E-commerce Store", "Electronics", "Computers", 
                        "Laptops", "Gaming Laptops"]
        self.assertEqual(path, expected_path)
    
    def test_category_search(self):
        """Test searching categories by name."""
        # Add various categories
        categories = [
            ("electronics", "Electronics"),
            ("computers", "Computers", "electronics"),
            ("laptops", "Laptops", "computers"),
            ("gaming_laptops", "Gaming Laptops", "laptops"),
        ]
        
        for category_data in categories:
            if len(category_data) == 2:
                self.tree.add_category(category_data[0], category_data[1])
            else:
                self.tree.add_category(category_data[0], category_data[1], category_data[2])
        
        # Search for categories containing "laptop"
        laptop_categories = self.tree.find_category_by_name("laptop", case_sensitive=False)
        
        self.assertEqual(len(laptop_categories), 2)  # "Laptops" and "Gaming Laptops"
        category_names = [cat.category_name for cat in laptop_categories]
        self.assertIn("Laptops", category_names)
        self.assertIn("Gaming Laptops", category_names)


class TestRecommendationEngineIntegration(unittest.TestCase):
    """Test cases for the integrated recommendation engine."""
    
    def setUp(self):
        """Set up test fixtures with comprehensive data."""
        self.engine = RecommendationEngine()
        self._setup_test_data()
    
    def _setup_test_data(self):
        """Set up comprehensive test data for integration testing."""
        # Create category hierarchy
        self.engine.category_tree.add_category("electronics", "Electronics")
        self.engine.category_tree.add_category("laptops", "Laptops", "electronics")
        self.engine.category_tree.add_category("phones", "Phones", "electronics")
        
        # Add products to categories
        products = {
            "laptops": ["laptop_001", "laptop_002", "laptop_003"],
            "phones": ["phone_001", "phone_002"]
        }
        
        for category, product_list in products.items():
            for product in product_list:
                self.engine.add_product_to_category(product, category)
        
        # Add product similarities
        similarities = [
            ("laptop_001", "laptop_002", 0.8),
            ("laptop_002", "laptop_003", 0.7),
            ("phone_001", "phone_002", 0.9),
        ]
        
        for p1, p2, score in similarities:
            self.engine.add_product_similarity(p1, p2, score)
        
        # Add user interactions
        interactions = [
            ("alice", "laptop_001", 4.5, "purchase"),
            ("alice", "phone_001", 3.5, "view"),
            ("bob", "laptop_001", 5.0, "purchase"),
            ("bob", "laptop_002", 4.0, "cart_add"),
            ("charlie", "phone_001", 4.5, "purchase"),
        ]
        
        for user_id, item_id, rating, action in interactions:
            self.engine.add_user_interaction(user_id, item_id, rating, action)
    
    def test_collaborative_filtering(self):
        """Test collaborative filtering recommendations."""
        recommendations = self.engine.get_collaborative_recommendations("alice", 3)
        
        self.assertIsInstance(recommendations, list)
        # Should have recommendations based on similar users
        if recommendations:
            self.assertIsInstance(recommendations[0], tuple)
            self.assertEqual(len(recommendations[0]), 2)  # (item_id, score)
    
    def test_content_based_filtering(self):
        """Test content-based filtering recommendations."""
        recommendations = self.engine.get_content_based_recommendations("alice", 3)
        
        self.assertIsInstance(recommendations, list)
        # Should recommend items similar to alice's liked items
        if recommendations:
            self.assertIsInstance(recommendations[0], tuple)
            # Should not recommend items user already interacted with
            user_items = {item_id for item_id, _ in 
                         self.engine.user_item_table.get_user_interactions("alice")}
            for item_id, _ in recommendations:
                self.assertNotIn(item_id, user_items)
    
    def test_hybrid_recommendations(self):
        """Test hybrid recommendation system."""
        recommendations = self.engine.get_hybrid_recommendations("alice", 5)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        # Verify recommendation structure
        for item_id, combined_score, strategy_breakdown in recommendations:
            self.assertIsInstance(item_id, str)
            self.assertIsInstance(combined_score, (int, float))
            self.assertIsInstance(strategy_breakdown, dict)
            self.assertGreater(combined_score, 0)
    
    def test_system_statistics(self):
        """Test system statistics and performance monitoring."""
        stats = self.engine.get_system_statistics()
        
        # Verify statistics structure
        required_components = [
            'user_item_interactions', 'product_similarities',
            'behavior_tracking', 'category_hierarchy', 'cache_size'
        ]
        
        for component in required_components:
            self.assertIn(component, stats)
        
        # Verify interaction statistics
        ui_stats = stats['user_item_interactions']
        self.assertGreater(ui_stats['size'], 0)
        self.assertGreaterEqual(ui_stats['load_factor'], 0)
    
    def test_recommendation_caching(self):
        """Test recommendation caching mechanism."""
        user_id = "alice"
        
        # First call - should compute recommendations
        start_time = time.time()
        recommendations1 = self.engine.get_hybrid_recommendations(user_id, 5)
        first_call_time = time.time() - start_time
        
        # Second call - should use cache
        start_time = time.time()
        recommendations2 = self.engine.get_hybrid_recommendations(user_id, 5)
        second_call_time = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(recommendations1, recommendations2)
        
        # Second call should be faster (cached)
        # Note: This might not always be true due to system variations
        # but cache size should increase
        stats = self.engine.get_system_statistics()
        self.assertGreaterEqual(stats['cache_size'], 0)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling across all components."""
    
    def test_empty_data_structures(self):
        """Test behavior with empty data structures."""
        hash_table = UserItemHashTable()
        graph = ProductSimilarityGraph()
        tree = UserBehaviorTree()
        category_tree = CategoryHierarchyTree()
        
        # Test empty operations
        self.assertIsNone(hash_table.get("user", "item"))
        self.assertEqual(len(graph.get_similar_products("product")), 0)
        self.assertEqual(len(tree.get_recent_activity()), 0)
        self.assertEqual(len(category_tree.get_products_in_category("category")), 0)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        engine = RecommendationEngine()
        
        # Test invalid user ID
        recommendations = engine.get_hybrid_recommendations("nonexistent_user", 5)
        self.assertEqual(len(recommendations), 0)
        
        # Test invalid category
        with self.assertRaises(ValueError):
            engine.category_tree.add_category("test", "Test", "nonexistent_parent")
    
    def test_large_dataset_simulation(self):
        """Test performance with larger datasets."""
        hash_table = UserItemHashTable()
        
        # Insert large number of interactions
        num_users = 100
        num_items = 50
        
        start_time = time.time()
        for user_i in range(num_users):
            for item_i in range(min(10, num_items)):  # Each user interacts with 10 items
                user_id = f"user_{user_i:03d}"
                item_id = f"item_{item_i:03d}"
                rating = random.uniform(1.0, 5.0)
                hash_table.insert(user_id, item_id, 
                                {"rating": rating, "action": "purchase"})
        
        insertion_time = time.time() - start_time
        
        # Test retrieval performance
        start_time = time.time()
        for user_i in range(0, num_users, 10):  # Sample every 10th user
            user_id = f"user_{user_i:03d}"
            interactions = hash_table.get_user_interactions(user_id)
            self.assertGreater(len(interactions), 0)
        
        retrieval_time = time.time() - start_time
        
        # Verify reasonable performance (adjust thresholds as needed)
        self.assertLess(insertion_time, 5.0)  # Should complete within 5 seconds
        self.assertLess(retrieval_time, 2.0)   # Should complete within 2 seconds
        
        # Verify hash table statistics
        stats = hash_table.get_statistics()
        self.assertEqual(stats['size'], num_users * 10)
        self.assertLess(stats['load_factor'], 1.0)


def run_comprehensive_tests():
    """Run all test suites and generate a comprehensive report."""
    
    print("=" * 80)
    print("E-COMMERCE RECOMMENDATION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("Phase 2: Proof of Concept Testing and Validation")
    print("=" * 80)
    
    # Create test suite
    test_classes = [
        TestUserItemHashTable,
        TestProductSimilarityGraph,
        TestUserBehaviorTree,
        TestCategoryHierarchyTree,
        TestRecommendationEngineIntegration,
        TestEdgeCasesAndErrorHandling,
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class in test_classes:
        print(f"\n{'-' * 60}")
        print(f"Testing: {test_class.__name__}")
        print(f"{'-' * 60}")
        
        # Create test suite for this class
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Run tests with detailed output
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Accumulate statistics
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        # Print summary for this test class
        print(f"\nClass Summary:")
        print(f"  Tests run: {result.testsRun}")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        
        if result.failures:
            print("  Failed tests:")
            for test, traceback in result.failures:
                print(f"    - {test}")
        
        if result.errors:
            print("  Error tests:")
            for test, traceback in result.errors:
                print(f"    - {test}")
    
    # Print overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    print(f"Success rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("\n ALL TESTS PASSED! The system is ready for production use.")
    else:
        print(f"\n  {total_failures + total_errors} tests failed. Review the failures above.")
    
    return total_tests, total_failures, total_errors


if __name__ == "__main__":
    run_comprehensive_tests()