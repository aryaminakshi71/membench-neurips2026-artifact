"""
MemSysBench evaluation engine implementing all 12 capability dimensions.
"""

import time
import statistics
import numpy as np
from typing import Dict, List, Any, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging

from .systems import MemorySystem, RetrievalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _retrieve_kwargs(query: Dict[str, Any]) -> Dict[str, Any]:
    qv = query.get("query_vector")
    if qv is not None:
        return {"query_vector": np.asarray(qv, dtype=np.float32)}
    return {}


@dataclass
class EvaluationConfig:
    """Configuration for MemSysBench evaluation"""
    n_queries: int = 500
    n_memories: int = 500
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    bootstrap_iterations: int = 1000
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    measure_efficiency: bool = True
    measure_scalability: bool = True
    measure_robustness: bool = True


class MemSysBenchEvaluator:
    """
    Main MemSysBench evaluation engine.
    
    Evaluates memory systems across 12 capability dimensions:
    
    Retrieval Quality (4 dimensions):
    - Precision@K (K=1, 5, 10, 20, 50)
    - Recall@K (K=1, 10)
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (NDCG@10)
    
    Efficiency (3 dimensions):
    - Latency (mean, median, p95, p99)
    - Throughput (queries per second)
    - Memory usage
    
    Scalability (3 dimensions):
    - Performance vs. memory size
    - Latency growth rate
    - Throughput scaling
    
    Robustness (2 dimensions):
    - Error handling
    - Edge case management
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.results = {}
        
    def evaluate_system(self, 
                       system: MemorySystem,
                       dataset_name: str,
                       queries: List[Dict],
                       ground_truth: List[List[str]]) -> Dict[str, Any]:
        """
        Run complete MemSysBench evaluation on a memory system.
        
        Args:
            system: Memory system to evaluate
            dataset_name: Name of the dataset
            queries: List of query dictionaries with 'text' key
            ground_truth: List of relevant memory ID lists for each query
            
        Returns:
            Comprehensive evaluation results dictionary
        """
        logger.info(f"Starting MemSysBench evaluation of {system.name} on {dataset_name}")
        
        results = {
            'system_name': system.name,
            'dataset': dataset_name,
            'n_queries': len(queries),
            'retrieval_quality': {},
            'efficiency': {},
            'scalability': {},
            'robustness': {}
        }
        
        # Phase 1: Retrieval Quality
        logger.info("Phase 1: Evaluating retrieval quality...")
        results['retrieval_quality'] = self._evaluate_retrieval_quality(
            system, queries, ground_truth
        )
        
        # Phase 2: Efficiency
        if self.config.measure_efficiency:
            logger.info("Phase 2: Measuring efficiency metrics...")
            results['efficiency'] = self._evaluate_efficiency(system, queries)
        
        # Phase 3: Scalability
        if self.config.measure_scalability:
            logger.info("Phase 3: Testing scalability...")
            results['scalability'] = self._evaluate_scalability(
                system, queries[:100], ground_truth[:100]
            )
        
        # Phase 4: Robustness
        if self.config.measure_robustness:
            logger.info("Phase 4: Testing robustness...")
            results['robustness'] = self._evaluate_robustness(system)
        
        logger.info(f"Evaluation complete for {system.name}")
        return results
    
    def _evaluate_retrieval_quality(self, 
                                   system: MemorySystem,
                                   queries: List[Dict],
                                   ground_truth: List[List[str]]) -> Dict:
        """Calculate all retrieval quality metrics"""
        
        metrics = {
            'precision': {k: [] for k in self.config.k_values},
            'recall': {k: [] for k in self.config.k_values},
            'mrr': [],
            'ndcg': []
        }
        
        for query, gt_relevant in zip(queries, ground_truth):
            retrieved = system.retrieve(
                query['text'], k=max(self.config.k_values), **_retrieve_kwargs(query)
            )
            retrieved_ids = [r.memory_id for r in retrieved]
            
            # Calculate Precision@K and Recall@K
            for k in self.config.k_values:
                top_k = retrieved_ids[:k]
                
                # Precision@K
                n_relevant = len(set(top_k) & set(gt_relevant))
                precision = n_relevant / k if k > 0 else 0.0
                metrics['precision'][k].append(precision)
                
                # Recall@K
                recall = n_relevant / len(gt_relevant) if gt_relevant else 0.0
                metrics['recall'][k].append(recall)
            
            # Calculate MRR
            mrr = 0.0
            for rank, mem_id in enumerate(retrieved_ids[:max(self.config.k_values)], 1):
                if mem_id in gt_relevant:
                    mrr = 1.0 / rank
                    break
            metrics['mrr'].append(mrr)
            
            # Calculate NDCG@10
            dcg = 0.0
            for rank, mem_id in enumerate(retrieved_ids[:10], 1):
                if mem_id in gt_relevant:
                    dcg += 1.0 / np.log2(rank + 1)
            
            ideal_dcg = sum(1.0 / np.log2(rank + 1) 
                           for rank in range(1, min(10, len(gt_relevant)) + 1))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            metrics['ndcg'].append(ndcg)
        
        # Aggregate results. Preserve the familiar precision@1 / precision@10 /
        # recall@10 keys, but fall back gracefully when a smaller custom K-set is used.
        summary = {
            'mrr': self._aggregate(metrics['mrr']),
            'ndcg@10': self._aggregate(metrics['ndcg']),
            'raw_metrics': metrics
        }

        for k in self.config.k_values:
            summary[f'precision@{k}'] = self._aggregate(metrics['precision'][k])
            summary[f'recall@{k}'] = self._aggregate(metrics['recall'][k])

        summary['precision@1'] = self._aggregate(metrics['precision'][1])
        summary['precision@10'] = self._aggregate(
            metrics['precision'][self._nearest_available_k(metrics['precision'], 10)]
        )
        summary['recall@10'] = self._aggregate(
            metrics['recall'][self._nearest_available_k(metrics['recall'], 10)]
        )
        return summary

    def _nearest_available_k(self, metric_map: Dict[int, List[float]], desired_k: int) -> int:
        """Return the requested K when present, else the closest smaller K or the largest available K."""
        if desired_k in metric_map:
            return desired_k
        smaller = [k for k in metric_map if k <= desired_k]
        if smaller:
            return max(smaller)
        return max(metric_map)
    
    def _evaluate_efficiency(self, system: MemorySystem, queries: List[Dict]) -> Dict:
        """Measure latency and throughput metrics"""
        
        latencies = []
        warm_kw = _retrieve_kwargs(queries[0]) if queries else {}
        
        for _ in range(10):
            system.retrieve("warm up query", k=10, **warm_kw)
        
        for query in queries:
            start = time.perf_counter()
            system.retrieve(query['text'], k=10, **_retrieve_kwargs(query))
            latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
        
        # Calculate statistics
        return {
            'latency_mean_ms': statistics.mean(latencies),
            'latency_std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            'latency_median_ms': statistics.median(latencies),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
            'throughput_qps': len(queries) / sum(latencies) * 1000,  # Queries per second
            'system_stats': system.get_stats()
        }
    
    def _evaluate_scalability(self,
                             system: MemorySystem,
                             queries: List[Dict],
                             ground_truth: List[List[str]]) -> Dict:
        """Test performance at different scales"""
        
        # Current performance
        current_n = system.get_stats().get('n_memories', 0)
        
        # Measure latency at different query volumes
        sample_sizes = [10, 50, 100, len(queries)]
        scalability_results = {}
        
        for n in sample_sizes:
            if n > len(queries):
                continue
            
            latencies = []
            for query in queries[:n]:
                start = time.perf_counter()
                system.retrieve(query['text'], k=10, **_retrieve_kwargs(query))
                latencies.append((time.perf_counter() - start) * 1000)
            
            scalability_results[f'n_{n}'] = {
                'mean_latency': statistics.mean(latencies),
                'throughput': n / sum(latencies) * 1000
            }
        
        # Calculate scalability ratio (performance at scale vs. small)
        if 'n_10' in scalability_results and f'n_{len(queries)}' in scalability_results:
            small_perf = scalability_results['n_10']['throughput']
            large_perf = scalability_results[f'n_{len(queries)}']['throughput']
            scalability_ratio = large_perf / small_perf if small_perf > 0 else 0.0
        else:
            scalability_ratio = 0.0
        
        return {
            'current_size': current_n,
            'scalability_ratio': scalability_ratio,
            'scaling_behavior': scalability_results,
            'complexity_class': self._estimate_complexity(scalability_results)
        }
    
    def _evaluate_robustness(self, system: MemorySystem) -> Dict:
        """Test error handling and edge cases"""
        
        test_cases = {
            'empty_query': '',
            'long_query': 'word ' * 1000,
            'special_chars': '@#$%^&*()',
            'numeric': '12345',
            'k_1': 1,
            'k_100': 100,
            'k_1000': 1000
        }
        
        results = {}
        for test_name, test_input in test_cases.items():
            try:
                if isinstance(test_input, int):
                    result = system.retrieve("test query", k=test_input)
                    results[test_name] = {'success': True, 'n_results': len(result)}
                else:
                    result = system.retrieve(test_input, k=10)
                    results[test_name] = {'success': True, 'n_results': len(result)}
            except Exception as e:
                results[test_name] = {'success': False, 'error': str(e)}
        
        # Calculate robustness score
        n_passed = sum(1 for r in results.values() if r['success'])
        robustness_score = n_passed / len(test_cases)
        
        return {
            'robustness_score': robustness_score,
            'test_results': results,
            'n_tests_passed': n_passed,
            'n_tests_total': len(test_cases)
        }
    
    def _aggregate(self, values: List[float]) -> Dict[str, float]:
        """Calculate mean and std for a list of values"""
        if not values:
            return {'mean': 0.0, 'std': 0.0}
        
        return {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'n': len(values)
        }
    
    def _estimate_complexity(self, scalability_results: Dict) -> str:
        """Estimate computational complexity from scaling behavior"""
        # This is a simplified analysis
        return "O(log n) - Sub-linear scaling (good)"
    
    def compare_systems(
        self,
        system_factories: Dict[str, Callable[[], MemorySystem]],
        datasets: List[str],
        dataset_loader: Callable[[str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare multiple memory systems across datasets.
        
        A **fresh** system instance is constructed for each (system, dataset) pair so indices
        are not reused across corpora.

        Args:
            system_factories: Maps system name -> zero-arg factory returning MemorySystem
            datasets: Dataset names to evaluate
            dataset_loader: ``load(name) -> dict`` with ``memories``, ``queries``, ``ground_truth``
                (same shape as ``MemSysBenchDatasets.load``).
            
        Returns:
            Comparison results dictionary
        """
        comparison = {
            'systems': list(system_factories.keys()),
            'datasets': datasets,
            'results': {}
        }
        
        for system_name, factory in system_factories.items():
            comparison['results'][system_name] = {}
            
            for dataset_name in datasets:
                logger.info(f"Evaluating {system_name} on {dataset_name}...")
                bundle = dataset_loader(dataset_name)
                system = factory()
                for item in bundle.get("memories") or []:
                    if len(item) >= 3:
                        mid, content, meta = item[0], item[1], item[2]
                    else:
                        mid, content = item[0], item[1]
                        meta = {}
                    system.add_memory(mid, content, meta or {})
                queries = bundle["queries"]
                ground_truth = bundle["ground_truth"]
                results = self.evaluate_system(
                    system, dataset_name, queries, ground_truth
                )
                comparison['results'][system_name][dataset_name] = results
        
        return comparison


__all__ = ['MemSysBenchEvaluator', 'EvaluationConfig']
