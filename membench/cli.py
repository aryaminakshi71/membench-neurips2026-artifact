"""
Command-line interface for MemBench.

Provides commands:
- membench evaluate: Evaluate a system on datasets
- membench compare: Compare multiple systems
- membench report: Generate reports from results
"""

import click
import json
from pathlib import Path
from typing import Dict

from .evaluator import MemBenchEvaluator, EvaluationConfig
from .datasets import MemBenchDatasets
from .builtin_systems import (
    BM25MemorySystem, TFIDFMemorySystem, DenseMemorySystem,
    FAISSFlatMemorySystem, HybridRRFMemorySystem
)
from .report import ReportGenerator


@click.group()
@click.version_option(version='1.0.0', prog_name='membench')
def cli():
    """
    MemBench: Comprehensive Memory System Evaluation Framework
    
    Evaluate and compare LLM memory systems across 12 capability dimensions
    using 5 standardized benchmark datasets.
    
    Example:
        membench evaluate --system faiss --dataset natural_questions
        membench compare --all --output comparison.html
    """
    pass


@cli.command()
@click.option('--system', 
              type=click.Choice(['bm25', 'tfidf', 'dense', 'faiss', 'hybrid'], 
                               case_sensitive=False),
              default='bm25',
              help='Memory system to evaluate')
@click.option('--dataset', '-d',
              multiple=True,
              default=['natural_questions'],
              help='Dataset(s) to evaluate on. Can specify multiple.')
@click.option('--all-datasets', is_flag=True, default=False,
              help='Run on every registered dataset (text + image embedding shards).')
@click.option('--n-memories', default=500, help='Number of memories to use')
@click.option('--n-queries', default=500, help='Number of queries to evaluate')
@click.option('--output', '-o', default='membench_results.json',
              help='Output file for results (JSON)')
@click.option('--report-format', 
              type=click.Choice(['latex', 'markdown', 'all']),
              default='all',
              help='Additional report formats to generate')
@click.option('--device', default='cpu', help='Device (cpu/cuda) for neural models')
@click.option(
    '--resume',
    is_flag=True,
    default=False,
    help='If --output exists, reuse prior results and only run datasets missing from that file.',
)
def evaluate(system, dataset, all_datasets, n_memories, n_queries, output, report_format, device, resume):
    """
    Evaluate a memory system on MemBench datasets.
    
    Example:
        membench evaluate --system bm25 --dataset natural_questions --n-memories 500
        membench evaluate --system dense -d ms_marco -d narrativeqa -o results.json
    """
    click.echo(f"🚀 MemBench Evaluation")
    click.echo(f"   System: {system.upper()}")
    ds_list = list(MemBenchDatasets.DATASET_CONFIGS.keys()) if all_datasets else list(dataset)
    click.echo(f"   Datasets: {', '.join(ds_list)}")
    click.echo(f"   Configuration: {n_memories} memories, {n_queries} queries\n")

    def _system_factory():
        if system == 'bm25':
            return BM25MemorySystem()
        if system == 'tfidf':
            return TFIDFMemorySystem()
        if system == 'dense':
            return DenseMemorySystem(device=device)
        if system == 'faiss':
            return FAISSFlatMemorySystem(device=device)
        return HybridRRFMemorySystem()

    # Initialize evaluator
    config = EvaluationConfig(n_memories=n_memories, n_queries=n_queries)
    evaluator = MemBenchEvaluator(config)
    
    # Run evaluation for each dataset
    out_path = Path(output)
    all_results = {
        'system': system,
        'config': {
            'n_memories': n_memories,
            'n_queries': n_queries
        },
        'results': {}
    }
    if resume and out_path.is_file():
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            if isinstance(prev.get('results'), dict):
                all_results['results'] = dict(prev['results'])
                click.echo(f"📂 Resume: loaded {len(all_results['results'])} dataset(s) from {output}")
        except Exception as e:
            click.echo(f"⚠️  Resume: could not read {output} ({e}); starting fresh.\n", err=True)

    for ds_name in ds_list:
        if resume and ds_name in all_results['results']:
            click.echo(f"📊 Skipping {ds_name} (already in {output})")
            continue
        click.echo(f"📊 Loading {ds_name}...")
        try:
            ds = MemBenchDatasets.load(ds_name, n_memories=n_memories, n_queries=n_queries)
            
            click.echo(f"   ✓ Loaded: {len(ds['memories'])} memories, {len(ds['queries'])} queries")

            mem_system = _system_factory()
            MemBenchDatasets.populate_system(mem_system, ds)

            click.echo(f"🔍 Running evaluation...")
            results = evaluator.evaluate_system(
                mem_system, ds_name, ds['queries'], ds['ground_truth']
            )
            
            all_results['results'][ds_name] = results
            
            # Print summary
            rq = results['retrieval_quality']
            eff = results['efficiency']
            click.echo(f"   ✓ MRR: {rq['mrr']['mean']:.4f} ± {rq['mrr']['std']:.4f}")
            click.echo(f"   ✓ Latency: {eff['latency_mean_ms']:.2f}ms")
            click.echo()
            
        except Exception as e:
            click.echo(f"   ✗ Error: {e}\n", err=True)
            continue
    
    # Save results
    with open(output, 'w') as f:
        json.dump(all_results, f, indent=2)
    click.echo(f"💾 Results saved to: {output}")
    
    # Generate additional reports
    if report_format in ['latex', 'all']:
        generator = ReportGenerator(all_results['results'])
        tables = generator.generate_latex_tables('./tables')
        click.echo(f"📄 LaTeX tables generated in ./tables/")
    
    if report_format in ['markdown', 'all']:
        generator = ReportGenerator(all_results['results'])
        generator.generate_markdown_summary('RESULTS.md')
        click.echo(f"📄 Markdown summary: RESULTS.md")
    
    click.echo("\n✅ Evaluation complete!")


@cli.command()
@click.option('--systems', '-s',
              default='bm25,tfidf,dense,faiss',
              help='Comma-separated list of systems to compare')
@click.option('--datasets', '-d',
              default='ms_marco,natural_questions,triviaqa',
              help='Comma-separated list of datasets')
@click.option('--all-datasets', is_flag=True, default=False,
              help='Compare on every registered dataset (text + image shards).')
@click.option('--n-memories', default=500)
@click.option('--n-queries', default=500)
@click.option('--output', '-o', default='comparison.json')
@click.option('--visualize/--no-visualize', default=True,
              help='Generate visualization plots')
def compare(systems, datasets, all_datasets, n_memories, n_queries, output, visualize):
    """
    Compare multiple memory systems across datasets.
    
    Example:
        membench compare --systems bm25,dense --datasets ms_marco,natural_questions
        membench compare -s bm25,tfidf,faiss --all-datasets -o full_comparison.json
    """
    system_list = [s.strip() for s in systems.split(',')]
    dataset_list = (
        list(MemBenchDatasets.DATASET_CONFIGS.keys())
        if all_datasets
        else [d.strip() for d in datasets.split(',') if d.strip()]
    )
    
    click.echo(f"🏆 MemBench Comparison")
    click.echo(f"   Systems: {', '.join(system_list)}")
    click.echo(f"   Datasets: {', '.join(dataset_list)}\n")
    
    factories = {}
    for sys_name in system_list:
        if sys_name == 'bm25':
            factories[sys_name] = BM25MemorySystem
        elif sys_name == 'tfidf':
            factories[sys_name] = TFIDFMemorySystem
        elif sys_name == 'dense':
            factories[sys_name] = lambda: DenseMemorySystem()
        elif sys_name == 'faiss':
            factories[sys_name] = lambda: FAISSFlatMemorySystem()
        elif sys_name == 'hybrid':
            factories[sys_name] = HybridRRFMemorySystem

    # Run comparison
    config = EvaluationConfig(n_memories=n_memories, n_queries=n_queries)
    evaluator = MemBenchEvaluator(config)
    
    click.echo("⏳ Running evaluations...\n")
    
    comparison = evaluator.compare_systems(
        factories,
        dataset_list,
        lambda ds: MemBenchDatasets.load(ds, n_memories=n_memories, n_queries=n_queries),
    )
    
    # Save results
    with open(output, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate reports
    generator = ReportGenerator(comparison)
    generator.print_summary()
    generator.generate_latex_tables('./comparison_tables')
    generator.generate_markdown_summary('COMPARISON.md')
    
    click.echo(f"\n💾 Results saved to: {output}")
    click.echo(f"📄 LaTeX tables: ./comparison_tables/")
    click.echo(f"📄 Summary: COMPARISON.md")
    click.echo("\n✅ Comparison complete!")


@cli.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--format', 'fmt',
              type=click.Choice(['latex', 'markdown', 'html', 'summary']),
              default='summary',
              help='Report format to generate')
@click.option('--output', '-o', help='Output file (defaults to appropriate extension)')
def report(results_file, fmt, output):
    """
    Generate reports from existing MemBench results.
    
    Example:
        membench report results.json --format latex
        membench report comparison.json --format markdown -o report.md
    """
    click.echo(f"📄 Generating {fmt} report from {results_file}...")
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Generate report
    generator = ReportGenerator(results)
    
    if fmt == 'latex':
        tables = generator.generate_latex_tables('./report_tables')
        click.echo(f"✅ LaTeX tables generated in ./report_tables/")
        
    elif fmt == 'markdown':
        out_file = output or 'REPORT.md'
        generator.generate_markdown_summary(out_file)
        click.echo(f"✅ Markdown report: {out_file}")
        
    elif fmt == 'summary':
        generator.print_summary()


@cli.command()
def list_datasets():
    """List available benchmark datasets."""
    click.echo("📚 Available MemBench Datasets:\n")
    
    datasets = MemBenchDatasets.list_datasets()
    for name, info in datasets.items():
        click.echo(f"  • {name}")
        click.echo(f"    Type: {info['type']}")
        click.echo(f"    Total samples: {info['n_total']:,}")
        click.echo()


@cli.command()
def info():
    """Show MemBench framework information."""
    click.echo("""
╔══════════════════════════════════════════════════════════════╗
║                    MemBench Framework v1.0.0                  ║
║                                                              ║
║  Comprehensive Evaluation Framework for LLM Memory Systems   ║
╚══════════════════════════════════════════════════════════════╝

📊 12 Capability Dimensions:
   Retrieval Quality: P@K, R@K, MRR, NDCG
   Efficiency: Latency, Throughput, Memory
   Scalability: Performance vs. Size, Growth Rate
   Robustness: Error Handling, Edge Cases

📚 5 Benchmark Datasets:
   • MS MARCO (82,226 passages)
   • NarrativeQA (32,647 passages)
   • NaturalQuestions (307,373 passages)
   • TriviaQA (87,522 passages)
   • PersonaChat (781,393 utterances)

🔧 Built-in Systems:
   • BM25 (sparse lexical)
   • TF-IDF (term frequency)
   • Dense-MiniLM (neural embeddings)
   • FAISS-Flat (exact similarity)
   • Hybrid-RRF (fusion)

📖 Documentation:
   GitHub: https://github.com/minakshi-arya/membench
   Paper: MemBench (ICML/NeurIPS 2026 submission)

Quick Start:
   membench evaluate --system bm25 --dataset natural_questions
   membench compare --systems bm25,dense,faiss --all-datasets
    """)


# Entry point
def main():
    """Entry point for CLI"""
    cli()


if __name__ == '__main__':
    main()


__all__ = ['cli', 'main']
