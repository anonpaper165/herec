"""
Core data structures for the structural leakage auditing framework.

An HESystem exposes its execution trace for each user query.
The auditor uses these traces to compute leakage metrics.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ExecutionTrace:
    """Server-observable metadata for one user query.

    Attributes:
        user_id:         user identifier
        true_degree:     actual number of interactions (ground truth, not
                         observable by server in practice — used for audit)
        n_chunks:        number of ciphertext chunks submitted
        chunk_indices:   which chunk indices were submitted (frozenset)
        n_rotations:     number of HE rotations scheduled (protocol-level)
        wall_time_ms:    measured wall-clock time in milliseconds
    """
    user_id: int
    true_degree: int
    n_chunks: int
    chunk_indices: frozenset
    n_rotations: int
    wall_time_ms: float = 0.0


@dataclass
class LeakageMetrics:
    """All leakage metrics for one system under one regime.

    Attributes:
        degree_mae:         mean absolute error in degree inference
                            (0 = perfect leak, higher = safer)
        reid_accuracy:      fraction of users correctly re-identified
                            (lower = safer)
        mean_anonymity_set: average size of anonymity set
                            (larger = safer)
        min_anonymity_set:  smallest anonymity set (worst-case user)
        timing_cv:          mean within-bucket coefficient of variation
                            of wall-clock time (lower = safer)
        chunk_pattern_unique_pct: fraction of users with a unique
                            chunk-index fingerprint within their bucket
                            (lower = safer)
        overhead:           latency blowup relative to unprotected sparse
    """
    degree_mae: float
    reid_accuracy: float
    mean_anonymity_set: float
    min_anonymity_set: int
    timing_cv: float
    chunk_pattern_unique_pct: float
    overhead: float = 1.0
    interaction_recovery_f1: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "degree_mae": self.degree_mae,
            "reid_accuracy": self.reid_accuracy,
            "mean_anonymity_set": self.mean_anonymity_set,
            "min_anonymity_set": self.min_anonymity_set,
            "timing_cv": self.timing_cv,
            "chunk_pattern_unique_pct": self.chunk_pattern_unique_pct,
            "overhead": self.overhead,
            "interaction_recovery_f1": self.interaction_recovery_f1,
        }


@dataclass
class AuditReport:
    """Full audit report for one system.

    Contains LeakageMetrics for each evaluated regime.
    """
    system_name: str
    protocol_type: str          # "interaction_indicator" or "embedding_query"
    dataset_name: str
    n_users_audited: int
    regimes: dict[str, LeakageMetrics] = field(default_factory=dict)

    def add(self, regime: str, metrics: LeakageMetrics):
        self.regimes[regime] = metrics

    def print(self):
        header = (f"\n{'='*70}\n"
                  f"  Structural Leakage Audit: {self.system_name}\n"
                  f"  Protocol: {self.protocol_type} | "
                  f"Dataset: {self.dataset_name} | "
                  f"N={self.n_users_audited}\n"
                  f"{'='*70}")
        print(header)
        col = "{:<18} {:>9} {:>9} {:>10} {:>10} {:>9} {:>8}"
        print(col.format(
            "Regime", "Deg.MAE", "Re-ID↓", "MeanAnon↑",
            "MinAnon↑", "TimCV↓", "Blowup"))
        print("  " + "-" * 66)
        for regime, m in self.regimes.items():
            print(col.format(
                regime,
                f"{m.degree_mae:.2f}",
                f"{m.reid_accuracy:.4f}",
                f"{m.mean_anonymity_set:.1f}",
                str(m.min_anonymity_set),
                f"{m.timing_cv:.3f}",
                f"{m.overhead:.2f}x",
            ))
        print(f"{'='*70}\n")

    def leakage_detected(self) -> bool:
        """True if the sparse regime has degree_mae == 0."""
        sparse = self.regimes.get("sparse")
        return sparse is not None and sparse.degree_mae == 0.0


class HESystem(ABC):
    """Abstract base class for an HE-based personalized inference system.

    Subclasses implement get_execution_trace() to expose what the server
    observes for a given user query. The auditor calls this method and
    computes leakage metrics from the returned traces.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable system name."""

    @property
    @abstractmethod
    def protocol_type(self) -> str:
        """'interaction_indicator' or 'embedding_query'."""

    @abstractmethod
    def get_execution_trace(self, user_id: int, regime: str = "sparse") -> ExecutionTrace:
        """Return the server-observable execution trace for user_id.

        Args:
            user_id: user to query
            regime: one of 'sparse', 'bucketed', 'decoy', 'dense'

        Returns:
            ExecutionTrace with all observable metadata filled in.
        """

    @abstractmethod
    def get_user_ids(self) -> list[int]:
        """Return all auditable user IDs."""
