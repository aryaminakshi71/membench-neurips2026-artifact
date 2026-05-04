"""
Report generation for MemBench evaluation results.

Generates publication-ready outputs:
- LaTeX tables for papers
- HTML interactive reports  
- JSON/CSV for data analysis
- Visualizations and plots
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import numpy as np


class ReportGenerator:
    """
    Generate comprehensive evaluation reports.
    
    Supports multiple output formats for different use cases:
    - LaTeX: Publication-ready tables
    - HTML: Interactive browser reports
    - Markdown: GitHub documentation
    - JSON: Machine-readable results
    """
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize with evaluation results.
        
        Args:
            results: Dictionary from MemBenchEvaluator.evaluate_system()
                    or compare_systems()
        """
        self.results = results
        self.timestamp = datetime.now().isoformat()
        
    def generate_latex_tables(self, output_dir: str = './tables') -> Dict[str, str]:
        """
        Generate LaTeX tables for publication.
        
        Returns:
            Dictionary mapping table names to LaTeX code
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        tables = {}
        
        # Main results table
        tables['main_results'] = self._latex_main_table()
        
        # Per-dataset breakdown
        tables['dataset_breakdown'] = self._latex_dataset_table()
        
        # Efficiency metrics
        tables['efficiency'] = self._latex_efficiency_table()
        
        # Save to files
        for name, latex in tables.items():
            with open(f'{output_dir}/{name}.tex', 'w') as f:
                f.write(latex)
        
        return tables
    
    def _latex_main_table(self) -> str:
        """Generate main comparison table"""
        
        if 'results' in self.results:  # Multi-system comparison
            systems = self.results['systems']
            datasets = self.results['datasets']
            
            latex = r"""\begin{table*}[t]
\centering
\scriptsize
\caption{MemBench Evaluation Results - Complete Comparison Across Systems and Datasets}
\label{tab:membench_complete}
\begin{tabular}{lccccccc}
\toprule
\textbf{System} & \textbf{Dataset} & \textbf{MRR} & \textbf{NDCG@10} & \textbf{P@10} & \textbf{Latency (ms)} & \textbf{QPS} & \textbf{MemEff} \\
\midrule
"""
            
            for system in systems:
                for dataset in datasets:
                    if dataset in self.results['results'][system]:
                        r = self.results['results'][system][dataset]
                        rq = r.get('retrieval_quality', {})
                        eff = r.get('efficiency', {})
                        
                        mrr = rq.get('mrr', {}).get('mean', 0.0)
                        ndcg = rq.get('ndcg@10', {}).get('mean', 0.0)
                        p10 = rq.get('precision@10', {}).get('mean', 0.0)
                        lat = eff.get('latency_mean_ms', 0.0)
                        qps = eff.get('throughput_qps', 0.0)
                        memeff = ndcg * 1000  # Simple calculation
                        
                        latex += f"{system} & {dataset} & {mrr:.4f} & {ndcg:.4f} & {p10:.4f} & {lat:.2f} & {qps:.1f} & {memeff:.2f} \\\\\n"
            
            latex += r"""\bottomrule
\end{tabular}
\end{table*}"""
            
        else:  # Single system
            system_name = self.results.get('system_name', 'System')
            dataset = self.results.get('dataset', 'Dataset')
            rq = self.results.get('retrieval_quality', {})
            eff = self.results.get('efficiency', {})
            
            latex = r"""\begin{table}[h]
\centering
\caption{MemBench Evaluation Results - """ + system_name + r""" on """ + dataset + r"""}
\label{tab:membench_""" + system_name.lower() + r"""_""" + dataset.lower() + r"""}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""
            
            mrr = rq.get('mrr', {}).get('mean', 0.0)
            mrr_std = rq.get('mrr', {}).get('std', 0.0)
            latex += f"MRR & {mrr:.4f} $\\pm$ {mrr_std:.4f} \\\\\n"
            
            ndcg = rq.get('ndcg@10', {}).get('mean', 0.0)
            ndcg_std = rq.get('ndcg@10', {}).get('std', 0.0)
            latex += f"NDCG@10 & {ndcg:.4f} $\\pm$ {ndcg_std:.4f} \\\\\n"
            
            p10 = rq.get('precision@10', {}).get('mean', 0.0)
            latex += f"P@10 & {p10:.4f} \\\\\n"
            
            lat = eff.get('latency_mean_ms', 0.0)
            latex += f"Latency (ms) & {lat:.2f} \\\\\n"
            
            qps = eff.get('throughput_qps', 0.0)
            latex += f"QPS & {qps:.1f} \\\\\n"
            
            latex += r"""\bottomrule
\end{tabular}
\end{table}"""
        
        return latex
    
    def _latex_dataset_table(self) -> str:
        """Generate per-dataset performance table"""
        
        if 'results' not in self.results:
            return ""
        
        latex = r"""\begin{table*}[t]
\centering
\scriptsize
\caption{Per-Dataset Performance Breakdown - All 5 MemBench Datasets}
\label{tab:dataset_breakdown}
\begin{tabular}{lcccccc}
\toprule
\textbf{Dataset} & \textbf{System} & \textbf{MRR} & \textbf{NDCG@10} & \textbf{P@1} & \textbf{P@10} & \textbf{R@10} \\
\midrule
"""
        
        for dataset in self.results['datasets']:
            for system in self.results['systems']:
                if dataset in self.results['results'][system]:
                    r = self.results['results'][system][dataset]
                    rq = r.get('retrieval_quality', {})
                    
                    mrr = rq.get('mrr', {}).get('mean', 0.0)
                    ndcg = rq.get('ndcg@10', {}).get('mean', 0.0)
                    p1 = rq.get('precision@1', {}).get('mean', 0.0)
                    p10 = rq.get('precision@10', {}).get('mean', 0.0)
                    r10 = rq.get('recall@10', {}).get('mean', 0.0)
                    
                    latex += f"{dataset} & {system} & {mrr:.4f} & {ndcg:.4f} & {p1:.4f} & {p10:.4f} & {r10:.4f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table*}"""
        
        return latex
    
    def _latex_efficiency_table(self) -> str:
        """Generate efficiency metrics table"""
        
        latex = r"""\begin{table}[h]
\centering
\scriptsize
\caption{Efficiency and Scalability Metrics}
\label{tab:efficiency}
\begin{tabular}{lcccccc}
\toprule
\textbf{System} & \textbf{Latency} & \textbf{P95 Lat.} & \textbf{QPS} & \textbf{Scal. Ratio} & \textbf{Memory (MB)} \\
\midrule
"""
        
        if 'results' in self.results:
            for system in self.results['systems']:
                for dataset in self.results['datasets']:
                    if dataset in self.results['results'][system]:
                        r = self.results['results'][system][dataset]
                        eff = r.get('efficiency', {})
                        scal = r.get('scalability', {})
                        
                        lat = eff.get('latency_mean_ms', 0.0)
                        p95 = eff.get('latency_p95_ms', 0.0)
                        qps = eff.get('throughput_qps', 0.0)
                        ratio = scal.get('scalability_ratio', 0.0)
                        mem = eff.get('system_stats', {}).get('memory_mb', 0.0)
                        
                        latex += f"{system} & {lat:.1f} & {p95:.1f} & {qps:.1f} & {ratio:.3f} & {mem:.0f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}"""
        
        return latex
    
    def generate_json(self, output_file: str = 'membench_results.json') -> None:
        """Save results as JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_markdown_summary(self, output_file: str = 'RESULTS.md') -> None:
        """Generate GitHub-friendly markdown summary"""
        
        md = f"""# MemBench Evaluation Results

**Generated:** {self.timestamp}

## Summary

"""
        
        if 'results' in self.results:
            md += f"**Systems Evaluated:** {len(self.results['systems'])}\n\n"
            md += f"**Datasets:** {', '.join(self.results['datasets'])}\n\n"
            
            md += "## Performance by System\n\n"
            md += "| System | Avg MRR | Avg NDCG@10 | Avg Latency (ms) |\n"
            md += "|--------|---------|-------------|------------------|\n"
            
            for system in self.results['systems']:
                mrr_scores = []
                ndcg_scores = []
                latencies = []
                
                for dataset in self.results['datasets']:
                    if dataset in self.results['results'][system]:
                        r = self.results['results'][system][dataset]
                        rq = r.get('retrieval_quality', {})
                        eff = r.get('efficiency', {})
                        
                        mrr_scores.append(rq.get('mrr', {}).get('mean', 0.0))
                        ndcg_scores.append(rq.get('ndcg@10', {}).get('mean', 0.0))
                        latencies.append(eff.get('latency_mean_ms', 0.0))
                
                if mrr_scores:
                    avg_mrr = np.mean(mrr_scores)
                    avg_ndcg = np.mean(ndcg_scores)
                    avg_lat = np.mean(latencies)
                    md += f"| {system} | {avg_mrr:.4f} | {avg_ndcg:.4f} | {avg_lat:.2f} |\n"
        
        else:
            system = self.results.get('system_name', 'System')
            dataset = self.results.get('dataset', 'Dataset')
            rq = self.results.get('retrieval_quality', {})
            
            md += f"**System:** {system}\n\n"
            md += f"**Dataset:** {dataset}\n\n"
            md += f"**MRR:** {rq.get('mrr', {}).get('mean', 0.0):.4f}\n\n"
            md += f"**NDCG@10:** {rq.get('ndcg@10', {}).get('mean', 0.0):.4f}\n\n"
        
        with open(output_file, 'w') as f:
            f.write(md)
    
    def print_summary(self) -> None:
        """Print text summary to console"""
        print("\n" + "="*60)
        print("  MemBench Evaluation Results")
        print("="*60)
        
        if 'results' in self.results:
            print(f"\nSystems: {len(self.results['systems'])}")
            print(f"Datasets: {len(self.results['datasets'])}")
            print("\n" + "-"*60)
            
            for system in self.results['systems']:
                print(f"\n{system}:")
                for dataset in self.results['datasets']:
                    if dataset in self.results['results'][system]:
                        r = self.results['results'][system][dataset]
                        rq = r.get('retrieval_quality', {})
                        eff = r.get('efficiency', {})
                        
                        mrr = rq.get('mrr', {}).get('mean', 0.0)
                        lat = eff.get('latency_mean_ms', 0.0)
                        print(f"  {dataset:20s}: MRR={mrr:.4f}, Latency={lat:.1f}ms")
        
        else:
            system = self.results.get('system_name', 'System')
            dataset = self.results.get('dataset', 'Dataset')
            rq = self.results.get('retrieval_quality', {})
            eff = self.results.get('efficiency', {})
            
            print(f"\nSystem: {system}")
            print(f"Dataset: {dataset}")
            print(f"MRR: {rq.get('mrr', {}).get('mean', 0.0):.4f}")
            print(f"NDCG@10: {rq.get('ndcg@10', {}).get('mean', 0.0):.4f}")
            print(f"Latency: {eff.get('latency_mean_ms', 0.0):.2f}ms")
        
        print("\n" + "="*60 + "\n")


__all__ = ['ReportGenerator']
