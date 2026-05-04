"""
Example: Compare multiple memory systems with MemBench.

This example shows how to compare different memory systems
(BM25, Dense, FAISS) across multiple datasets.
"""

from membench import MemBenchEvaluator, MemBenchDatasets
from membench.builtin_systems import (
    BM25MemorySystem, TFIDFMemorySystem, DenseMemorySystem,
    FAISSFlatMemorySystem
)
from membench.report import ReportGenerator


def main():
    print("=" * 70)
    print("MemBench Example: Comparing Multiple Memory Systems")
    print("=" * 70)
    
    # Define systems to compare
    print("\n[1/5] Setting up systems...")
    systems = {
        'BM25': BM25MemorySystem(),
        'TF-IDF': TFIDFMemorySystem(),
        'Dense': DenseMemorySystem(model_name='all-MiniLM-L6-v2'),
        # Note: FAISS requires more memory
        # 'FAISS': FAISSFlatMemorySystem(),
    }
    
    for name in systems:
        print(f"   ✓ {name}")
    
    # Define datasets
    print("\n[2/5] Selecting datasets...")
    datasets = ['ms_marco', 'natural_questions']
    for ds in datasets:
        print(f"   ✓ {ds}")
    
    # Initialize evaluator
    print("\n[3/5] Initializing evaluator...")
    from membench.evaluator import EvaluationConfig
    config = EvaluationConfig(n_memories=500, n_queries=500)
    evaluator = MemBenchEvaluator(config)
    
    # Run comparison
    print("\n[4/5] Running comparisons (this may take a few minutes)...")
    print()
    
    comparison_results = {}
    
    for system_name, system in systems.items():
        print(f"Evaluating {system_name}...")
        comparison_results[system_name] = {}
        
        for dataset_name in datasets:
            print(f"  Loading {dataset_name}...", end=' ')
            dataset = MemBenchDatasets.load(
                dataset_name,
                n_memories=500,
                n_queries=500
            )
            print(f"({len(dataset['memories'])} memories, {len(dataset['queries'])} queries)")
            
            # Add memories
            for mem_id, content, metadata in dataset['memories']:
                system.add_memory(mem_id, content, metadata)
            
            # Evaluate
            print(f"  Evaluating...", end=' ')
            results = evaluator.evaluate_system(
                system, dataset_name,
                dataset['queries'], dataset['ground_truth']
            )
            
            rq = results['retrieval_quality']
            eff = results['efficiency']
            print(f"MRR={rq['mrr']['mean']:.3f}, Latency={eff['latency_mean_ms']:.1f}ms")
            
            comparison_results[system_name][dataset_name] = results
    
    # Generate comparison report
    print("\n[5/5] Generating comparison report...")
    
    comparison = {
        'systems': list(systems.keys()),
        'datasets': datasets,
        'results': comparison_results
    }
    
    generator = ReportGenerator(comparison)
    generator.print_summary()
    
    # Save reports
    generator.generate_markdown_summary('comparison_results.md')
    tables = generator.generate_latex_tables('./comparison_tables')
    
    print("\n" + "=" * 70)
    print("Output files:")
    print("  📄 comparison_results.md - Markdown summary")
    print("  📄 comparison_tables/   - LaTeX tables for papers")
    print("=" * 70)


if __name__ == '__main__':
    main()
