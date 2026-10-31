"""Heterogeneous work graph and local Personalized PageRank.

Nodes are users, items, projects and tags. Edges come from authorship, assignment,
interaction strength, project membership, tagging, parent and child relations and explicit
links between items. Personalized PageRank from a seed set (a user, a task, or both) gives
a relevance score to every reachable node that captures multi hop structure such as
"documents linked from tasks in projects you work on" or "people who completed tasks that
look like this one".

The push algorithm (Andersen, Chung and Lang, 2006) only touches the neighbourhood of the
seed, so per request cost is bounded by the tolerance rather than by graph size.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

USER = "u"
ITEM = "i"
PROJECT = "p"
TAG = "t"


def node_key(kind: str, identifier: str) -> str:
    return f"{kind}:{identifier}"


@dataclass
class GraphBuilder:
    edges: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))

    def add(self, a: str, b: str, weight: float) -> None:
        if a == b or weight <= 0:
            return
        self.edges[(a, b)] += weight
        self.edges[(b, a)] += weight

    def build(self) -> WorkGraph:
        nodes = sorted({n for pair in self.edges for n in pair})
        index = {n: i for i, n in enumerate(nodes)}
        if not nodes:
            return WorkGraph([], {}, sparse.csr_matrix((0, 0), dtype=np.float32))
        rows = np.fromiter((index[a] for a, _ in self.edges), dtype=np.int64, count=len(self.edges))
        cols = np.fromiter((index[b] for _, b in self.edges), dtype=np.int64, count=len(self.edges))
        data = np.fromiter(self.edges.values(), dtype=np.float32, count=len(self.edges))
        adj = sparse.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
        return WorkGraph(nodes, index, adj)


class WorkGraph:
    def __init__(self, nodes: list[str], index: dict[str, int], adjacency: sparse.csr_matrix) -> None:
        self.nodes = nodes
        self.index = index
        self.adjacency = adjacency.tocsr()
        degree = np.asarray(self.adjacency.sum(axis=1)).ravel().astype(np.float32)
        degree[degree == 0] = 1.0
        self.degree = degree

    @property
    def size(self) -> int:
        return len(self.nodes)

    def has(self, key: str) -> bool:
        return key in self.index

    def personalized_pagerank(
        self,
        seeds: dict[str, float],
        damping: float = 0.85,
        epsilon: float = 1e-5,
        max_pushes: int = 200_000,
    ) -> dict[str, float]:
        """Approximate PPR by residual push. Returns node key to score for touched nodes."""
        seed_idx = {self.index[k]: w for k, w in seeds.items() if k in self.index and w > 0}
        if not seed_idx:
            return {}
        total = sum(seed_idx.values())
        alpha = 1.0 - damping
        residual: dict[int, float] = {i: w / total for i, w in seed_idx.items()}
        estimate: dict[int, float] = defaultdict(float)
        frontier = list(residual)
        indptr, indices, data = self.adjacency.indptr, self.adjacency.indices, self.adjacency.data
        pushes = 0
        while frontier and pushes < max_pushes:
            node = frontier.pop()
            r = residual.get(node, 0.0)
            deg = self.degree[node]
            if r < epsilon * deg:
                continue
            pushes += 1
            estimate[node] += alpha * r
            # Lazy walk variant keeps half the mass at the node for stability.
            remaining = (1.0 - alpha) * r
            residual[node] = remaining / 2.0
            spread = remaining / 2.0
            start, end = indptr[node], indptr[node + 1]
            for j in range(start, end):
                neighbour = int(indices[j])
                share = spread * float(data[j]) / deg
                before = residual.get(neighbour, 0.0)
                after = before + share
                residual[neighbour] = after
                if before < epsilon * self.degree[neighbour] <= after:
                    frontier.append(neighbour)
            if residual[node] >= epsilon * deg:
                frontier.append(node)
        return {self.nodes[i]: float(v) for i, v in estimate.items() if v > 0}

    def neighbours(self, key: str) -> list[tuple[str, float]]:
        idx = self.index.get(key)
        if idx is None:
            return []
        start, end = self.adjacency.indptr[idx], self.adjacency.indptr[idx + 1]
        return [
            (self.nodes[int(self.adjacency.indices[j])], float(self.adjacency.data[j]))
            for j in range(start, end)
        ]
