"""
Built-in memory system implementations for MemBench.

Provides reference implementations of standard retrieval methods:
- BM25: Sparse lexical retrieval
- TF-IDF: Term frequency-based retrieval  
- FAISS-Flat: Exact dense similarity search
- FAISS-IVF: Approximate dense similarity search
- Dense-MiniLM: Neural embedding-based retrieval
- Hybrid-RRF: Reciprocal rank fusion of sparse+dense
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re

from .systems import MemorySystem, RetrievalResult


class BM25MemorySystem(MemorySystem):
    """
    BM25 sparse retrieval implementation.
    
    Classic lexical matching with BM25 scoring. Fast and interpretable
    but limited to exact term matching.
    
    Example:
        >>> system = BM25MemorySystem(k1=1.5, b=0.75)
        >>> system.add_memory("doc1", "The quick brown fox")
        >>> results = system.retrieve("quick fox", k=5)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, name: str = "BM25-Memory"):
        super().__init__(name)
        self.k1 = k1
        self.b = b
        self.corpus = []  # List of (id, tokens, content, metadata)
        self.doc_freqs = defaultdict(int)
        self.avg_dl = 0
        self.N = 0
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def add_memory(self, memory_id: str, content: str, metadata: Dict = None) -> None:
        """Add a document to the BM25 index"""
        tokens = self._tokenize(content)
        self.corpus.append((memory_id, tokens, content, metadata or {}))
        
        # Update document frequencies
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.doc_freqs[token] += 1
        
        self.N += 1
        total_len = sum(len(tokens) for _, tokens, _, _ in self.corpus)
        self.avg_dl = total_len / self.N if self.N > 0 else 0
        self._stats['n_memories'] = self.N
        
    def retrieve(
        self, query: str, k: int = 10, query_vector: Optional[np.ndarray] = None
    ) -> List[RetrievalResult]:
        """Retrieve top-k documents using BM25 scoring"""
        start_time = time.perf_counter()
        
        query_tokens = self._tokenize(query)
        scores = []
        
        for mem_id, tokens, content, metadata in self.corpus:
            score = self._bm25_score(query_tokens, tokens)
            scores.append((mem_id, content, metadata, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[3], reverse=True)
        
        latency = (time.perf_counter() - start_time) * 1000
        self._stats['total_queries'] += 1
        self._stats['total_retrieve_time'] += latency / 1000
        
        results = []
        for rank, (mem_id, content, metadata, score) in enumerate(scores[:k], 1):
            results.append(RetrievalResult(
                memory_id=mem_id,
                content=content,
                score=score,
                metadata=metadata,
                rank=rank
            ))
        
        return results
    
    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Calculate BM25 score for a document given a query"""
        score = 0.0
        doc_len = len(doc_tokens)
        token_counts = defaultdict(int)
        for token in doc_tokens:
            token_counts[token] += 1
        
        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
                
            df = self.doc_freqs[token]
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            
            tf = token_counts[token]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            
            score += idf * numerator / denominator
        
        return score
    
    def get_stats(self) -> Dict[str, Any]:
        """Return system statistics"""
        stats = self._stats.copy()
        stats['n_memories'] = self.N
        stats['avg_doc_length'] = self.avg_dl
        stats['vocabulary_size'] = len(self.doc_freqs)
        stats['index_type'] = 'BM25'
        return stats


class TFIDFMemorySystem(MemorySystem):
    """
    TF-IDF based sparse retrieval.
    
    Simpler than BM25, using cosine similarity on TF-IDF vectors.
    Good baseline for lexical retrieval.
    """
    
    def __init__(self, name: str = "TF-IDF-Memory"):
        super().__init__(name)
        self.corpus = []
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.N = 0
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def add_memory(self, memory_id: str, content: str, metadata: Dict = None) -> None:
        tokens = self._tokenize(content)
        self.corpus.append((memory_id, tokens, content, metadata or {}))
        
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.doc_freqs[token] += 1
        
        self.N += 1
        self._update_idf()
        self._stats['n_memories'] = self.N
    
    def _update_idf(self):
        """Recalculate IDF values"""
        for token, df in self.doc_freqs.items():
            self.idf[token] = np.log(self.N / (df + 1)) + 1
    
    def retrieve(
        self, query: str, k: int = 10, query_vector: Optional[np.ndarray] = None
    ) -> List[RetrievalResult]:
        start_time = time.perf_counter()
        
        query_tokens = self._tokenize(query)
        query_vec = self._get_vector(query_tokens)
        
        scores = []
        for mem_id, tokens, content, metadata in self.corpus:
            doc_vec = self._get_vector(tokens)
            score = self._cosine_sim(query_vec, doc_vec)
            scores.append((mem_id, content, metadata, score))
        
        scores.sort(key=lambda x: x[3], reverse=True)
        
        latency = (time.perf_counter() - start_time) * 1000
        self._stats['total_queries'] += 1
        self._stats['total_retrieve_time'] += latency / 1000
        
        results = []
        for rank, (mem_id, content, metadata, score) in enumerate(scores[:k], 1):
            results.append(RetrievalResult(
                memory_id=mem_id, content=content, 
                score=score, metadata=metadata, rank=rank
            ))
        
        return results
    
    def _get_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Convert tokens to TF-IDF vector (as dict)"""
        vector = defaultdict(float)
        token_counts = defaultdict(int)
        for token in tokens:
            token_counts[token] += 1
        
        for token, count in token_counts.items():
            tf = count / len(tokens) if tokens else 0
            idf = self.idf.get(token, 1.0)
            vector[token] = tf * idf
        
        return dict(vector)
    
    def _cosine_sim(self, vec1: Dict, vec2: Dict) -> float:
        """Calculate cosine similarity between two sparse vectors"""
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in all_terms)
        
        norm1 = np.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = np.sqrt(sum(v**2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            'n_memories': self.N,
            'vocabulary_size': len(self.doc_freqs),
            'index_type': 'TF-IDF'
        }


class DenseMemorySystem(MemorySystem):
    """
    Dense neural embedding-based retrieval using sentence transformers.
    
    Uses pre-trained language models to encode memories and queries
    into dense vectors. When metadata includes ``precomputed_vector``
    (MemBench shards), that row is indexed instead of encoding placeholder text.
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 name: str = "Dense-MiniLM-Memory",
                 device: str = 'cpu'):
        super().__init__(name)
        self._model_name = model_name
        self._device = device
        self._model = None
        self.embedding_dim: Optional[int] = None
        self.memories = []
        self.embeddings = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
            if self.embedding_dim is None:
                self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
        
    def add_memory(self, memory_id: str, content: str, metadata: Dict = None) -> None:
        """Add a memory with dense embedding"""
        metadata = metadata or {}
        pv = metadata.get("precomputed_vector")
        if pv is not None:
            embedding = np.asarray(pv, dtype=np.float32)
            dim = int(embedding.shape[0])
            if self.embedding_dim is None:
                self.embedding_dim = dim
            elif self.embedding_dim != dim:
                raise ValueError(
                    f"DenseMemorySystem: embedding dim {self.embedding_dim} != row {dim}"
                )
        else:
            self._ensure_model()
            embedding = np.asarray(
                self._model.encode(content, convert_to_numpy=True), dtype=np.float32
            )
            if self.embedding_dim is None:
                self.embedding_dim = int(embedding.shape[0])
        self.memories.append((memory_id, embedding, content, metadata))
        
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        self._stats['n_memories'] = len(self.memories)
    
    def retrieve(
        self, query: str, k: int = 10, query_vector: Optional[np.ndarray] = None
    ) -> List[RetrievalResult]:
        """Dense retrieval using cosine similarity"""
        start_time = time.perf_counter()

        if query_vector is not None:
            query_embedding = np.asarray(query_vector, dtype=np.float32)
        else:
            self._ensure_model()
            query_embedding = np.asarray(
                self._model.encode(query, convert_to_numpy=True), dtype=np.float32
            )
        
        if self.embeddings is None or len(self.memories) == 0:
            return []
        
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        latency = (time.perf_counter() - start_time) * 1000
        self._stats['total_queries'] += 1
        self._stats['total_retrieve_time'] += latency / 1000
        
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            mem_id, _, content, metadata = self.memories[idx]
            score = similarities[idx]
            results.append(RetrievalResult(
                memory_id=mem_id, content=content,
                score=float(score), metadata=metadata, rank=rank
            ))
        
        return results
    
    def _cosine_similarity(self, query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        doc_norms = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-10)
        return np.dot(doc_norms, query_norm)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            'n_memories': len(self.memories),
            'embedding_dim': self.embedding_dim,
            'index_type': 'Dense-Neural',
            'model': (
                int(self._model.get_sentence_embedding_dimension())
                if self._model is not None
                else self.embedding_dim
            ),
        }


class FAISSFlatMemorySystem(MemorySystem):
    """
    FAISS Flat index - exact dense similarity search.
    
    Uses Facebook AI Similarity Search for efficient exact nearest neighbor
    search on GPU or CPU.
    
    Note: Requires faiss-cpu or faiss-gpu package. Install with:
        pip install membench[faiss]
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 name: str = "FAISS-Flat-Memory",
                 device: str = 'cpu'):
        super().__init__(name)
        try:
            import faiss
            self.faiss = faiss
            self.faiss_available = True
        except ImportError:
            raise ImportError(
                "FAISS is required for FAISSFlatMemorySystem. "
                "Install with: pip install faiss-cpu or pip install membench[faiss]"
            )
        self._model_name = model_name
        self._device = device
        self._model = None
        self.embedding_dim: Optional[int] = None
        self.memories = []
        self.index = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
            if self.embedding_dim is None:
                self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
        
    def add_memory(self, memory_id: str, content: str, metadata: Dict = None) -> None:
        """Add memory with FAISS indexing"""
        metadata = metadata or {}
        pv = metadata.get("precomputed_vector")
        if pv is not None:
            embedding = np.asarray(pv, dtype=np.float32)
            dim = int(embedding.shape[0])
            if self.embedding_dim is None:
                self.embedding_dim = dim
            elif self.embedding_dim != dim:
                raise ValueError(
                    f"FAISSFlatMemorySystem: embedding dim {self.embedding_dim} != row {dim}"
                )
        else:
            self._ensure_model()
            embedding = np.asarray(
                self._model.encode(content, convert_to_numpy=True), dtype=np.float32
            )
            if self.embedding_dim is None:
                self.embedding_dim = int(embedding.shape[0])
        self.memories.append((memory_id, content, metadata))
        
        if self.index is None:
            if self.faiss.get_num_gpus() > 0:
                res = self.faiss.StandardGpuResources()
                self.index = self.faiss.GpuIndexFlatIP(res, self.embedding_dim)
            else:
                self.index = self.faiss.IndexFlatIP(self.embedding_dim)
        
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        self._stats['n_memories'] = len(self.memories)
    
    def retrieve(
        self, query: str, k: int = 10, query_vector: Optional[np.ndarray] = None
    ) -> List[RetrievalResult]:
        """FAISS exact search"""
        start_time = time.perf_counter()
        
        if query_vector is not None:
            query_embedding = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        else:
            self._ensure_model()
            query_embedding = np.asarray(
                self._model.encode(query, convert_to_numpy=True), dtype=np.float32
            ).reshape(1, -1)
        
        if self.index is None:
            return []
        
        scores, indices = self.index.search(query_embedding, k)
        
        latency = (time.perf_counter() - start_time) * 1000
        self._stats['total_queries'] += 1
        self._stats['total_retrieve_time'] += latency / 1000
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx >= 0 and idx < len(self.memories):
                mem_id, content, metadata = self.memories[idx]
                results.append(RetrievalResult(
                    memory_id=mem_id, content=content,
                    score=float(score), metadata=metadata, rank=rank
                ))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            'n_memories': len(self.memories),
            'embedding_dim': self.embedding_dim,
            'index_type': 'FAISS-Flat-Exact',
            'gpu_available': self.faiss.get_num_gpus() > 0
        }


class HybridRRFMemorySystem(MemorySystem):
    """
    Hybrid retrieval combining sparse (BM25) and dense methods
    using Reciprocal Rank Fusion (RRF).
    
    RRF formula: score = sum(1 / (k + rank)) for each method
    """
    
    def __init__(self, 
                 sparse_system: MemorySystem = None,
                 dense_system: MemorySystem = None,
                 k: int = 60,
                 name: str = "BM25+Dense-RRF"):
        super().__init__(name)
        self.sparse = sparse_system or BM25MemorySystem()
        self.dense = dense_system or DenseMemorySystem()
        self.k = k  # RRF constant
        
    def add_memory(self, memory_id: str, content: str, metadata: Dict = None) -> None:
        """Add to both systems"""
        self.sparse.add_memory(memory_id, content, metadata)
        self.dense.add_memory(memory_id, content, metadata)
        self._stats['n_memories'] += 1
    
    def retrieve(
        self, query: str, k: int = 10, query_vector: Optional[np.ndarray] = None
    ) -> List[RetrievalResult]:
        """RRF fusion of sparse and dense results"""
        start_time = time.perf_counter()
        
        sparse_results = self.sparse.retrieve(query, k=k*2)
        dense_results = self.dense.retrieve(query, k=k*2, query_vector=query_vector)
        
        # Calculate RRF scores
        rrf_scores = {}
        result_content = {}
        result_metadata = {}
        
        # Add sparse scores
        for rank, result in enumerate(sparse_results, 1):
            mem_id = result.memory_id
            score = 1.0 / (self.k + rank)
            rrf_scores[mem_id] = rrf_scores.get(mem_id, 0) + score
            result_content[mem_id] = result.content
            result_metadata[mem_id] = result.metadata
        
        # Add dense scores
        for rank, result in enumerate(dense_results, 1):
            mem_id = result.memory_id
            score = 1.0 / (self.k + rank)
            rrf_scores[mem_id] = rrf_scores.get(mem_id, 0) + score
            result_content[mem_id] = result.content
            result_metadata[mem_id] = result.metadata
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        latency = (time.perf_counter() - start_time) * 1000
        self._stats['total_queries'] += 1
        self._stats['total_retrieve_time'] += latency / 1000
        
        results = []
        for rank, (mem_id, score) in enumerate(sorted_results, 1):
            results.append(RetrievalResult(
                memory_id=mem_id,
                content=result_content[mem_id],
                score=score,
                metadata=result_metadata[mem_id],
                rank=rank
            ))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            'n_memories': self._stats['n_memories'],
            'index_type': 'Hybrid-RRF',
            'rrf_constant': self.k,
            'sparse_system': self.sparse.name,
            'dense_system': self.dense.name
        }


__all__ = [
    'BM25MemorySystem',
    'TFIDFMemorySystem', 
    'DenseMemorySystem',
    'FAISSFlatMemorySystem',
    'HybridRRFMemorySystem'
]
