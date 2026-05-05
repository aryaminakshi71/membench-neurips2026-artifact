# MemSysBench Supplementary Material

This supplementary package contains the anonymized MemSysBench framework used for the submitted benchmark paper.

## Contents

- `memsysbench/`: installable Python package with dataset loading, backend interfaces, evaluator, CLI, and report generation.
- `examples/`: minimal examples for evaluating a baseline, comparing systems, and defining a custom memory system.
- `tests/`: framework tests for evaluator and backend behavior.
- `tables/`: generated result tables used by the paper.
- `requirements.txt` and `requirements_pinned.txt`: runtime dependency specifications.
- `setup.py`: package installation metadata.

## Installation

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

FAISS-backed runs additionally require a FAISS installation, for example `faiss-cpu`.

## Basic Use

Evaluate one backend on one dataset:

```bash
memsysbench evaluate --system faiss --dataset natural_questions --n-memories 500 --n-queries 500
```

Compare multiple backends:

```bash
memsysbench compare --systems bm25,tfidf,faiss --datasets ms_marco,natural_questions
```

## Programmatic Use

```python
from memsysbench import MemSysBenchDatasets, MemSysBenchEvaluator
from memsysbench.systems import FAISSFlatMemorySystem

bundle = MemSysBenchDatasets.load("natural_questions", n_memories=500, n_queries=500)
system = FAISSFlatMemorySystem()
MemSysBenchDatasets.populate_system(system, bundle)

evaluator = MemSysBenchEvaluator()
results = evaluator.evaluate_system(
    system,
    "natural_questions",
    bundle["queries"],
    bundle["ground_truth"],
)
```

## Adding a Dataset

New datasets can be used by producing a MemSysBench bundle with:

- `memories`: `(memory_id, content, metadata)` tuples
- `queries`: dictionaries with query identifiers, text or image references, and optional precomputed vectors
- `ground_truth`: relevant memory identifiers for each query
- `info`: dataset metadata

Once this bundle is available, the same backend adapters, evaluator, metric computation, and report writer can be reused.

## Reproducibility Notes

The submitted paper reports runs over fixed public datasets and precomputed embedding shards. Large raw datasets and embedding shards are not included in this ZIP. The loader supports local embedding roots through `LLMRESEARCH_EMBEDDINGS_DIR` or `LLMRESEARCH_SHARED_DATASETS_DIR`.
