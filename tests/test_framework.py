#!/usr/bin/env python3
"""
Test script to verify MemSysBench framework functionality.
Run this to ensure the framework is working correctly.
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        import memsysbench
        from memsysbench.systems import MemorySystem, RetrievalResult
        from memsysbench.builtin_systems import (
            BM25MemorySystem, TFIDFMemorySystem, DenseMemorySystem
        )
        from memsysbench.evaluator import MemSysBenchEvaluator, EvaluationConfig
        from memsysbench.datasets import MemSysBenchDatasets
        from memsysbench.report import ReportGenerator
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_system_creation():
    """Test creating memory systems"""
    print("\nTesting system creation...")
    try:
        from memsysbench.builtin_systems import (
            BM25MemorySystem, TFIDFMemorySystem
        )
        
        # Test BM25
        bm25 = BM25MemorySystem()
        assert bm25.name == "BM25-Memory"
        print("  ✓ BM25 system created")
        
        # Test TF-IDF
        tfidf = TFIDFMemorySystem()
        assert tfidf.name == "TF-IDF-Memory"
        print("  ✓ TF-IDF system created")
        
        print("✓ All systems created successfully")
        return True
    except Exception as e:
        print(f"✗ System creation failed: {e}")
        traceback.print_exc()
        return False


def test_memory_operations():
    """Test adding and retrieving memories"""
    print("\nTesting memory operations...")
    try:
        from memsysbench.builtin_systems import BM25MemorySystem
        
        system = BM25MemorySystem()
        
        # Add memories
        memories = [
            ("doc1", "The quick brown fox jumps over the lazy dog", {}),
            ("doc2", "Machine learning is a subset of artificial intelligence", {}),
            ("doc3", "Python is a programming language", {}),
        ]
        
        for mem_id, content, meta in memories:
            system.add_memory(mem_id, content, meta)
        
        print(f"  ✓ Added {len(memories)} memories")
        
        # Retrieve
        results = system.retrieve("python programming", k=2)
        assert len(results) > 0, "No results returned"
        print(f"  ✓ Retrieved {len(results)} results")
        
        # Check result structure
        result = results[0]
        assert hasattr(result, 'memory_id')
        assert hasattr(result, 'content')
        assert hasattr(result, 'score')
        print("  ✓ Result structure valid")
        
        print("✓ Memory operations successful")
        return True
    except Exception as e:
        print(f"✗ Memory operations failed: {e}")
        traceback.print_exc()
        return False


def test_evaluator_creation():
    """Test creating evaluator"""
    print("\nTesting evaluator creation...")
    try:
        from memsysbench.evaluator import MemSysBenchEvaluator, EvaluationConfig
        
        config = EvaluationConfig(n_queries=10, n_memories=10)
        evaluator = MemSysBenchEvaluator(config)
        
        print("  ✓ Evaluator created")
        print("✓ Evaluator creation successful")
        return True
    except Exception as e:
        print(f"✗ Evaluator creation failed: {e}")
        traceback.print_exc()
        return False


def test_report_generator():
    """Test report generation"""
    print("\nTesting report generation...")
    try:
        from memsysbench.report import ReportGenerator
        
        # Create mock results
        mock_results = {
            'system_name': 'TestSystem',
            'dataset': 'test_dataset',
            'retrieval_quality': {
                'mrr': {'mean': 0.5, 'std': 0.1},
                'ndcg@10': {'mean': 0.6, 'std': 0.1}
            },
            'efficiency': {
                'latency_mean_ms': 10.0,
                'throughput_qps': 100.0
            }
        }
        
        generator = ReportGenerator(mock_results)
        
        # Test summary generation
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        generator.print_summary()
        sys.stdout = old_stdout
        
        print("  ✓ Summary generated")
        print("✓ Report generation successful")
        return True
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        traceback.print_exc()
        return False


def test_cli_import():
    """Test CLI import"""
    print("\nTesting CLI...")
    try:
        from memsysbench.cli import cli
        print("  ✓ CLI imported")
        print("✓ CLI import successful")
        return True
    except Exception as e:
        print(f"✗ CLI import failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation_flow():
    """Test complete evaluation flow with minimal data"""
    print("\nTesting complete evaluation flow...")
    try:
        from memsysbench.builtin_systems import BM25MemorySystem
        from memsysbench.evaluator import MemSysBenchEvaluator, EvaluationConfig
        
        # Create small system
        system = BM25MemorySystem()
        
        # Add test memories
        test_memories = [
            ("m1", "Machine learning is fascinating", {"type": "ml"}),
            ("m2", "Deep learning uses neural networks", {"type": "dl"}),
            ("m3", "Python programming is fun", {"type": "python"}),
            ("m4", "Natural language processing with transformers", {"type": "nlp"}),
            ("m5", "Computer vision applications", {"type": "cv"}),
        ]
        
        for mem_id, content, meta in test_memories:
            system.add_memory(mem_id, content, meta)
        
        # Create test queries and ground truth
        queries = [
            {"text": "machine learning neural networks"},
            {"text": "python programming"},
        ]
        ground_truth = [
            ["m1", "m2"],  # ml, dl
            ["m3"],        # python
        ]
        
        # Evaluate
        config = EvaluationConfig(n_queries=2, n_memories=5, k_values=[1, 2, 5])
        evaluator = MemSysBenchEvaluator(config)
        
        results = evaluator.evaluate_system(
            system, 'test', queries, ground_truth
        )
        
        # Verify results structure
        assert 'retrieval_quality' in results
        assert 'efficiency' in results
        assert 'mrr' in results['retrieval_quality']
        
        print("  ✓ Evaluation completed")
        print(f"  ✓ MRR: {results['retrieval_quality']['mrr']['mean']:.4f}")
        print("✓ Complete evaluation flow successful")
        return True
        
    except Exception as e:
        print(f"✗ Complete evaluation flow failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("MemSysBench Framework Test Suite")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("System Creation", test_system_creation),
        ("Memory Operations", test_memory_operations),
        ("Evaluator Creation", test_evaluator_creation),
        ("Report Generation", test_report_generator),
        ("CLI Import", test_cli_import),
        ("Complete Evaluation Flow", test_evaluation_flow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Framework is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
