"""
Example: Evaluate a custom memory system with MemSysBench.

This example shows how to implement and evaluate your own
memory system using the MemSysBench framework.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from memsysbench.systems import MemorySystem, RetrievalResult
from memsysbench import MemSysBenchEvaluator, MemSysBenchDatasets
from memsysbench.report import ReportGenerator


class MyCustomMemorySystem(MemorySystem):
    """
    Example custom memory system implementation.
    
    This is a simple dense retrieval system with cosine similarity.
    You can replace this with your own implementation.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(name="MyCustomSystem")
        self.model = SentenceTransformer(model_name)
        self.memories = []  # List of (id, embedding, content, metadata)
        
    def add_memory(self, memory_id: str, content: str, metadata: Dict = None) -> None:
        """Add a memory with embedding"""
        embedding = self.model.encode(content, convert_to_numpy=True)
        self.memories.append((memory_id, embedding, content, metadata or {}))
        
    def retrieve(
        self, query: str, k: int = 10, query_vector: Optional[np.ndarray] = None
    ) -> List[RetrievalResult]:
        """Retrieve using cosine similarity"""
        import time
        start_time = time.perf_counter()
        
        if query_vector is not None:
            query_emb = np.asarray(query_vector, dtype=np.float32)
        else:
            query_emb = self.model.encode(query, convert_to_numpy=True)
        
        # Calculate similarities
        results = []
        for mem_id, mem_emb, content, metadata in self.memories:
            # Cosine similarity
            sim = np.dot(query_emb, mem_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(mem_emb)
            )
            results.append((mem_id, content, metadata, float(sim)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[3], reverse=True)
        
        latency = (time.perf_counter() - start_time) * 1000
        self._stats['total_queries'] += 1
        self._stats['total_retrieve_time'] += latency / 1000
        
        # Return top-k
        return [
            RetrievalResult(
                memory_id=mem_id,
                content=content,
                score=score,
                metadata=metadata,
                rank=rank
            )
            for rank, (mem_id, content, metadata, score) in enumerate(results[:k], 1)
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'n_memories': len(self.memories),
            'embedding_dim': self.model.get_sentence_embedding_dimension(),
            'index_type': 'Custom-Dense'
        }


def main():
    print("=" * 70)
    print("MemSysBench Example: Evaluating Your Custom Memory System")
    print("=" * 70)
    
    print("\n📝 Step-by-step guide:")
    print("   1. Subclass MemorySystem")
    print("   2. Implement add_memory(), retrieve(), get_stats()")
    print("   3. Pass to MemSysBenchEvaluator")
    print()
    
    # Step 1: Initialize your custom system
    print("[1/3] Initializing custom memory system...")
    my_system = MyCustomMemorySystem(model_name='all-MiniLM-L6-v2')
    print(f"   System: {my_system.name}")
    print(f"   Embedding dim: {my_system.model.get_sentence_embedding_dimension()}")
    
    # Step 2: Load dataset
    print("\n[2/3] Loading MS MARCO dataset...")
    dataset = MemSysBenchDatasets.load('ms_marco', n_memories=500, n_queries=500)
    print(f"   Loaded: {len(dataset['memories'])} memories, {len(dataset['queries'])} queries")
    
    # Step 3: Add memories
    print("\n[3/3] Indexing memories...")
    for i, (mem_id, content, metadata) in enumerate(dataset['memories']):
        my_system.add_memory(mem_id, content, metadata)
        if (i + 1) % 100 == 0:
            print(f"   Added {i + 1}/{len(dataset['memories'])} memories...")
    
    print(f"\n   ✓ Total indexed: {len(my_system.memories)} memories")
    
    # Step 4: Evaluate
    print("\n[4/4] Running MemSysBench evaluation...")
    evaluator = MemSysBenchEvaluator()
    results = evaluator.evaluate_system(
        my_system,
        dataset_name='ms_marco',
        queries=dataset['queries'],
        ground_truth=dataset['ground_truth']
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results for Your Custom System")
    print("=" * 70)
    
    rq = results['retrieval_quality']
    eff = results['efficiency']
    scal = results['scalability']
    
    print(f"\n📊 Retrieval Quality:")
    print(f"   MRR:         {rq['mrr']['mean']:.4f} ± {rq['mrr']['std']:.4f}")
    print(f"   NDCG@10:     {rq['ndcg@10']['mean']:.4f} ± {rq['ndcg@10']['std']:.4f}")
    print(f"   Precision@1: {rq['precision@1']['mean']:.4f}")
    print(f"   Precision@10: {rq['precision@10']['mean']:.4f}")
    
    print(f"\n⚡ Efficiency:")
    print(f"   Mean Latency:  {eff['latency_mean_ms']:.2f} ms")
    print(f"   P95 Latency:   {eff['latency_p95_ms']:.2f} ms")
    print(f"   Throughput:    {eff['throughput_qps']:.1f} QPS")
    
    print(f"\n📈 Scalability:")
    print(f"   Scalability Ratio: {scal['scalability_ratio']:.3f}")
    print(f"   Complexity Class:  {scal['complexity_class']}")
    
    # Generate report
    print("\n📄 Generating evaluation report...")
    generator = ReportGenerator({'ms_marco': results})
    generator.generate_markdown_summary('my_system_results.md')
    generator.generate_latex_tables('./my_system_tables')
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Check my_system_results.md for detailed results")
    print("  2. Use LaTeX tables in ./my_system_tables/ for papers")
    print("  3. Compare with other systems using memsysbench compare")
    print("=" * 70)


if __name__ == '__main__':
    main()
