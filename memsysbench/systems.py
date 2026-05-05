"""
Base memory system interface and built-in implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import time
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Result from a memory retrieval operation"""
    memory_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int


class MemorySystem(ABC):
    """
    Abstract base class for all memory systems to be evaluated by MemSysBench.
    
    To evaluate your own memory system:
    1. Subclass MemorySystem
    2. Implement all abstract methods
    3. Pass to MemSysBenchEvaluator
    
    Example:
        class MyMemorySystem(MemorySystem):
            def add_memory(self, memory_id, content, metadata=None):
                # Your implementation
                pass
                
            def retrieve(self, query, k=10):
                # Your implementation
                return results
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self._stats = {
            'n_memories': 0,
            'total_add_time': 0.0,
            'total_queries': 0,
            'total_retrieve_time': 0.0
        }
    
    @abstractmethod
    def add_memory(self, memory_id: str, content: str, metadata: Dict = None) -> None:
        """
        Add a memory to the system.
        
        Args:
            memory_id: Unique identifier for the memory
            content: Text content of the memory
            metadata: Optional metadata dictionary
        """
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 10,
        query_vector: Optional[np.ndarray] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k memories relevant to the query.
        
        Args:
            query: Query text
            k: Number of results to return
            query_vector: Optional precomputed query embedding (same space as indexed memories)
            
        Returns:
            List of RetrievalResult objects, sorted by relevance (highest first)
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Return system statistics for evaluation.
        
        Returns:
            Dictionary containing:
                - n_memories: Total number of stored memories
                - memory_size_mb: Estimated memory usage in MB
                - index_type: Type of retrieval index used
                - embedding_dim: Dimension of embeddings (if applicable)
        """
        pass
    
    def batch_add(self, memories: List[Tuple[str, str, Optional[Dict]]]) -> None:
        """
        Add multiple memories efficiently.
        
        Args:
            memories: List of (memory_id, content, metadata) tuples
        """
        start_time = time.time()
        for mem_id, content, meta in memories:
            self.add_memory(mem_id, content, meta)
        
        self._stats['n_memories'] += len(memories)
        self._stats['total_add_time'] += time.time() - start_time
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Return latency statistics from operations"""
        stats = self._stats.copy()
        if stats['total_queries'] > 0:
            stats['avg_query_latency'] = stats['total_retrieve_time'] / stats['total_queries']
        return stats


# Import built-in systems
try:
    from .builtin_systems import (
        BM25MemorySystem,
        TFIDFMemorySystem,
        FAISSFlatMemorySystem,
        DenseMemorySystem,
        HybridRRFMemorySystem
    )
    
    __all__ = [
        'MemorySystem',
        'RetrievalResult',
        'BM25MemorySystem',
        'TFIDFMemorySystem',
        'FAISSFlatMemorySystem',
        'DenseMemorySystem',
        'HybridRRFMemorySystem'
    ]
except ImportError:
    # Fallback if optional dependencies are missing
    __all__ = ['MemorySystem', 'RetrievalResult']
