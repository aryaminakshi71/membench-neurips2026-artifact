"""
Example: Evaluate a built-in memory system with MemBench.

This example shows how to evaluate one of the built-in systems
(BM25, TF-IDF, Dense, FAISS) on a MemBench dataset.
"""

from membench import MemBenchEvaluator, MemBenchDatasets
from membench.builtin_systems import BM25MemorySystem, DenseMemorySystem
from membench.report import ReportGenerator


def main():
    print("=" * 60)
    print("MemBench Example: Evaluating Built-in Memory Systems")
    print("=" * 60)
    
    # Step 1: Initialize a memory system
    print("\n[1/4] Initializing memory system...")
    system = BM25MemorySystem(k1=1.5, b=0.75)
    # Alternative: system = DenseMemorySystem(model_name='all-MiniLM-L6-v2')
    print(f"   System: {system.name}")
    
    # Step 2: Load benchmark dataset
    print("\n[2/4] Loading NaturalQuestions dataset...")
    dataset = MemBenchDatasets.load(
        'natural_questions',
        n_memories=500,
        n_queries=500
    )
    print(f"   Memories: {len(dataset['memories'])}")
    print(f"   Queries: {len(dataset['queries'])}")
    
    # Step 3: Add memories to the system
    print("\n[3/4] Adding memories to system...")
    for mem_id, content, metadata in dataset['memories']:
        system.add_memory(mem_id, content, metadata)
    print(f"   Added {system.get_stats()['n_memories']} memories")
    
    # Step 4: Run evaluation
    print("\n[4/4] Running MemBench evaluation...")
    evaluator = MemBenchEvaluator()
    results = evaluator.evaluate_system(
        system=system,
        dataset_name='natural_questions',
        queries=dataset['queries'],
        ground_truth=dataset['ground_truth']
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    rq = results['retrieval_quality']
    eff = results['efficiency']
    
    print(f"\nRetrieval Quality:")
    print(f"  MRR:        {rq['mrr']['mean']:.4f} ± {rq['mrr']['std']:.4f}")
    print(f"  NDCG@10:    {rq['ndcg@10']['mean']:.4f} ± {rq['ndcg@10']['std']:.4f}")
    print(f"  P@1:        {rq['precision@1']['mean']:.4f}")
    print(f"  P@10:       {rq['precision@10']['mean']:.4f}")
    
    print(f"\nEfficiency:")
    print(f"  Latency:    {eff['latency_mean_ms']:.2f} ms (mean)")
    print(f"  P95 Latency: {eff['latency_p95_ms']:.2f} ms")
    print(f"  Throughput: {eff['throughput_qps']:.1f} QPS")
    
    # Generate reports
    print("\n[Bonus] Generating reports...")
    generator = ReportGenerator({'natural_questions': results})
    generator.generate_markdown_summary('example_results.md')
    tables = generator.generate_latex_tables('./example_tables')
    print("  ✓ Markdown report: example_results.md")
    print("  ✓ LaTeX tables: ./example_tables/")
    
    print("\n" + "=" * 60)
    print("Example complete! 🎉")
    print("=" * 60)


if __name__ == '__main__':
    main()
