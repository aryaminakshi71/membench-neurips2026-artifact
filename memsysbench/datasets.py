"""
Standardized dataset loaders for MemSysBench (precomputed embeddings only).

Loads vectors from ``local/shared_datasets/embeddings/text`` (or
``LLMRESEARCH_EMBEDDINGS_DIR``). Each row is a memory; the first ``n_queries``
rows are also used as queries with ground truth equal to the matching memory id
(same placeholder string as the stored memory), so dense/BM25 runs complete without
Hugging Face or raw corpora.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .systems import MemorySystem

import numpy as np

logger = logging.getLogger(__name__)

try:
    from shared.utils.precomputed_embeddings import IMAGE_DATASETS as _IMAGE_SLUGS
except ImportError:
    _IMAGE_SLUGS = (
        "all_idb",
        "aml",
        "blood_cancer",
        "blood_cell_all",
        "bmc",
        "cnmc",
        "idb",
        "multi_cancer",
    )

# Slug in filenames under embeddings/text/
_SLUG_BY_NAME = {
    "ms_marco": "ms_marco",
    "narrativeqa": "narrativeqa",
    "natural_questions": "natural_questions",
    "triviaqa": "triviaqa",
    "persona_chat": "personachat",
}


def _embeddings_root() -> Path:
    env = os.environ.get("LLMRESEARCH_EMBEDDINGS_DIR", "").strip()
    if env:
        p = Path(env)
        return p if p.name == "embeddings" else p / "embeddings"
    shared = os.environ.get("LLMRESEARCH_SHARED_DATASETS_DIR", "").strip()
    if shared:
        return Path(shared) / "embeddings"
    return Path(__file__).resolve().parents[2] / "shared_datasets" / "embeddings"


def _embeddings_shard_dir(modality: str) -> Path:
    sub = "text" if modality == "text" else "image"
    root = _embeddings_root()
    if os.environ.get("LLMRESEARCH_EMBEDDINGS_DIR", "").strip():
        e = Path(os.environ["LLMRESEARCH_EMBEDDINGS_DIR"].strip())
        if e.name == sub:
            return e
        return e / sub
    return root / sub


def _pick_embedding_files(slug: str, model: str, modality: str) -> Tuple[Path, Path]:
    base = _embeddings_shard_dir(modality)
    emb = base / f"{slug}_{model}_train.npy"
    meta = base / f"{slug}_{model}_meta.json"
    return emb, meta


def _fallback_text_models(_slug: str, preferred: str) -> List[str]:
    candidates = [
        preferred,
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    ]
    seen = set()
    ordered: List[str] = []
    for m in candidates:
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _fallback_image_models(preferred: str) -> List[str]:
    candidates = [preferred, "resnet50", "densenet121"]
    seen = set()
    ordered: List[str] = []
    for m in candidates:
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


class MemSysBenchDatasets:
    """
    Load benchmark corpora from precomputed embedding shards (no Hugging Face).
    """

    DATASET_CONFIGS = {
        "ms_marco": {"n_total": 82226, "type": "passage_ranking"},
        "narrativeqa": {"n_total": 32647, "type": "narrative_qa"},
        "natural_questions": {"n_total": 307373, "type": "factual_qa"},
        "triviaqa": {"n_total": 87522, "type": "trivia_qa"},
        "persona_chat": {"n_total": 781393, "type": "conversational"},
        **{slug: {"n_total": 50000, "type": "image_embedding"} for slug in _IMAGE_SLUGS},
    }

    @classmethod
    def load(
        cls,
        name: str,
        n_memories: int = 500,
        n_queries: int = 500,
    ) -> Dict[str, Any]:
        if name not in cls.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {name}. Available: {list(cls.DATASET_CONFIGS.keys())}"
            )

        slug = _SLUG_BY_NAME.get(name, name)
        modality = "image" if slug in _IMAGE_SLUGS else "text"
        if modality == "image":
            preferred = os.environ.get("LLMRESEARCH_IMAGE_EMBEDDING_MODEL", "resnet50").strip()
            model_candidates = _fallback_image_models(preferred)
        else:
            preferred = os.environ.get("LLMRESEARCH_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
            model_candidates = _fallback_text_models(slug, preferred)
        emb_path: Optional[Path] = None
        meta_path: Optional[Path] = None
        used_model: Optional[str] = None
        for model in model_candidates:
            e, m = _pick_embedding_files(slug, model, modality)
            if e.is_file() and m.is_file():
                emb_path, meta_path, used_model = e, m, model
                break

        if emb_path is None or meta_path is None:
            raise FileNotFoundError(
                f"No precomputed embeddings for '{name}' (slug={slug}) under {_embeddings_shard_dir(modality)}. "
                f"Expected files like {slug}_<model>_train.npy and _meta.json. "
                f"Set LLMRESEARCH_EMBEDDINGS_DIR to the embeddings root if needed."
            )

        arr = np.load(emb_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        n_rows = int(arr.shape[0])
        M = min(n_memories, n_rows)
        Q = min(n_queries, M)

        logger.info(
            "Loaded %s from precomputed embeddings (%s, %s rows, using model=%s)",
            name,
            emb_path.name,
            n_rows,
            used_model,
        )

        memories: List[Tuple[str, str, Dict[str, Any]]] = []
        for i in range(M):
            mid = f"{slug}_{i}"
            txt = f"[precomputed_embedding:{slug}:{i}]"
            vec = np.asarray(arr[i], dtype=np.float32)
            memories.append(
                (
                    mid,
                    txt,
                    {
                        "row": i,
                        "embedding_model": used_model,
                        "precomputed_vector": vec,
                    },
                )
            )

        queries: List[Dict[str, Any]] = []
        ground_truth: List[List[str]] = []
        for q in range(Q):
            mid = f"{slug}_{q}"
            qvec = np.asarray(arr[q], dtype=np.float32)
            queries.append(
                {
                    "id": f"q_{q}",
                    "text": f"[precomputed_embedding:{slug}:{q}]",
                    "query_vector": qvec,
                }
            )
            ground_truth.append([mid])

        cfg = cls.DATASET_CONFIGS[name]
        return {
            "memories": memories,
            "queries": queries,
            "ground_truth": ground_truth,
            "info": {
                "name": name,
                "type": cfg["type"],
                "source": "precomputed_embeddings",
                "embedding_file": str(emb_path),
                "meta": meta,
                "embedding_model": used_model,
            },
        }

    @staticmethod
    def populate_system(mem_system: MemorySystem, bundle: Dict[str, Any]) -> None:
        """
        Insert all memories from a load() bundle into a MemorySystem before evaluate_system().

        MemSysBench evaluators expect the system index to be built from ``bundle["memories"]``,
        each item is ``(memory_id, content, metadata_dict)``.
        """
        for item in bundle.get("memories") or []:
            if len(item) == 3:
                mid, content, meta = item
            else:
                mid, content = item[0], item[1]
                meta = {}
            mem_system.add_memory(mid, content, meta or {})

    @classmethod
    def list_datasets(cls) -> Dict[str, Dict[str, Any]]:
        return {
            name: {"n_total": config["n_total"], "type": config["type"]}
            for name, config in cls.DATASET_CONFIGS.items()
        }


__all__ = ["MemSysBenchDatasets"]
