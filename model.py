"""Per tenant recommender model: training, persistence and an atomic in memory registry."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import sparse

from vantix.config import Settings
from vantix.embeddings.base import l2_normalise
from vantix.observability.logging import get_logger
from vantix.observability.metrics import MODEL_TRAININGS
from vantix.recommender.als import ALSConfig, ImplicitALS
from vantix.recommender.graph import ITEM, PROJECT, TAG, USER, GraphBuilder, WorkGraph, node_key
from vantix.storage.repository import Repository

log = get_logger(__name__)

_ROLE_WEIGHTS = {"author": 3.0, "assignee": 3.0}


def decay(age_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return float(0.5 ** (max(0.0, age_days) / half_life_days))


@dataclass
class TenantModel:
    tenant_id: str
    trained_at: datetime
    user_ids: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    als: ImplicitALS | None = None
    graph: WorkGraph | None = None
    popularity: dict[str, float] = field(default_factory=dict)
    activity: dict[str, float] = field(default_factory=dict)
    expertise_users: list[str] = field(default_factory=list)
    expertise: np.ndarray | None = None
    user_index: dict[str, int] = field(default_factory=dict)
    item_index: dict[str, int] = field(default_factory=dict)
    expertise_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reindex()

    def reindex(self) -> None:
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {it: i for i, it in enumerate(self.item_ids)}
        self.expertise_index = {u: i for i, u in enumerate(self.expertise_users)}

    @classmethod
    def empty(cls, tenant_id: str) -> TenantModel:
        return cls(tenant_id=tenant_id, trained_at=datetime.fromtimestamp(0, UTC))


class RecommenderTrainer:
    def __init__(self, settings: Settings, repository: Repository, embedding_dim: int) -> None:
        self.settings = settings
        self.repo = repository
        self.dim = embedding_dim

    def train(self, tenant_id: str, now: datetime | None = None) -> TenantModel:
        s = self.settings
        now = now or datetime.now(UTC)
        try:
            weights: dict[tuple[str, str], float] = defaultdict(float)
            popularity: dict[str, float] = defaultdict(float)
            activity: dict[str, float] = defaultdict(float)
            for user_id, item_id, _type, weight, occurred_at in self.repo.iter_interactions(tenant_id):
                if weight <= 0:
                    continue
                age = (now - occurred_at).total_seconds() / 86400.0
                weights[(user_id, item_id)] += weight * decay(age, s.rec_interaction_half_life_days)
                popularity[item_id] += weight * decay(age, s.rec_popularity_half_life_days)
                activity[user_id] += decay(age, s.rec_popularity_half_life_days)

            users_by_item: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for (user_id, item_id), w in weights.items():
                users_by_item[item_id].append((user_id, w))

            graph = GraphBuilder()
            for (user_id, item_id), w in weights.items():
                graph.add(node_key(USER, user_id), node_key(ITEM, item_id), float(np.log1p(w)))

            expertise_acc: dict[str, np.ndarray] = {}
            expertise_mass: dict[str, float] = defaultdict(float)

            def accumulate(user: str, vector: np.ndarray, w: float) -> None:
                if user not in expertise_acc:
                    expertise_acc[user] = np.zeros(self.dim, dtype=np.float32)
                expertise_acc[user] += w * vector
                expertise_mass[user] += w

            for stored in self.repo.iter_items(tenant_id):
                item = stored.item
                ik = node_key(ITEM, item.id)
                if item.project_id:
                    graph.add(ik, node_key(PROJECT, item.project_id), 1.0)
                for tag in item.tags:
                    graph.add(ik, node_key(TAG, tag), 0.5)
                if item.author_id:
                    graph.add(ik, node_key(USER, item.author_id), _ROLE_WEIGHTS["author"])
                for assignee in item.assignee_ids:
                    graph.add(ik, node_key(USER, assignee), _ROLE_WEIGHTS["assignee"])
                for linked in item.linked_item_ids:
                    graph.add(ik, node_key(ITEM, linked), 1.5)
                if item.parent_id:
                    graph.add(ik, node_key(ITEM, item.parent_id), 1.5)
                vec = stored.embedding
                if vec is not None and vec.shape[0] == self.dim:
                    if item.author_id:
                        accumulate(item.author_id, vec, _ROLE_WEIGHTS["author"])
                    for assignee in item.assignee_ids:
                        accumulate(assignee, vec, _ROLE_WEIGHTS["assignee"])
                    for user_id, w in users_by_item.get(item.id, []):
                        accumulate(user_id, vec, float(np.log1p(w)))

            user_ids = sorted({u for u, _ in weights})
            item_ids = sorted({i for _, i in weights})
            als: ImplicitALS | None = None
            if len(user_ids) >= 2 and len(item_ids) >= 2 and len(weights) >= 5:
                u_index = {u: n for n, u in enumerate(user_ids)}
                i_index = {i: n for n, i in enumerate(item_ids)}
                rows = [u_index[u] for u, _ in weights]
                cols = [i_index[i] for _, i in weights]
                matrix = sparse.csr_matrix(
                    (np.fromiter(weights.values(), dtype=np.float32), (rows, cols)),
                    shape=(len(user_ids), len(item_ids)),
                )
                factors = max(2, min(s.rec_als_factors, len(user_ids), len(item_ids)))
                als = ImplicitALS(
                    ALSConfig(
                        factors=factors,
                        regularization=s.rec_als_regularization,
                        iterations=s.rec_als_iterations,
                        alpha=s.rec_als_alpha,
                    )
                ).fit(matrix)

            expertise_users = sorted(expertise_acc)
            expertise = (
                l2_normalise(np.vstack([expertise_acc[u] for u in expertise_users]))
                if expertise_users
                else None
            )
            model = TenantModel(
                tenant_id=tenant_id,
                trained_at=now,
                user_ids=user_ids if als else [],
                item_ids=item_ids if als else [],
                als=als,
                graph=graph.build(),
                popularity=_normalise_dict(popularity),
                activity=_normalise_dict(activity),
                expertise_users=expertise_users,
                expertise=expertise,
            )
            self.repo.set_model_trained(tenant_id, now)
            MODEL_TRAININGS.labels(tenant=tenant_id, outcome="success").inc()
            log.info(
                "recommender_trained",
                extra={
                    "tenant": tenant_id,
                    "users": len(user_ids),
                    "items": len(item_ids),
                    "graph_nodes": model.graph.size if model.graph else 0,
                    "als": als is not None,
                },
            )
            return model
        except Exception:
            MODEL_TRAININGS.labels(tenant=tenant_id, outcome="failure").inc()
            raise


def _normalise_dict(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    top = max(values.values())
    if top <= 0:
        return {}
    return {k: v / top for k, v in values.items()}


class ModelRegistry:
    """Holds the current model per tenant and persists models to disk for warm restarts."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._models: dict[str, TenantModel] = {}
        self._lock = threading.RLock()

    def get(self, tenant_id: str) -> TenantModel:
        with self._lock:
            model = self._models.get(tenant_id)
        if model is None:
            model = self.load(tenant_id) or TenantModel.empty(tenant_id)
            with self._lock:
                self._models.setdefault(tenant_id, model)
                model = self._models[tenant_id]
        return model

    def put(self, model: TenantModel, persist: bool = True) -> None:
        with self._lock:
            self._models[model.tenant_id] = model
        if persist:
            self.save(model)

    def _path(self, tenant_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.]", "_", tenant_id)
        return self.model_dir / safe

    def save(self, model: TenantModel) -> None:
        final = self._path(model.tenant_id)
        tmp = final.with_name(final.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        meta = {
            "tenant_id": model.tenant_id,
            "trained_at": model.trained_at.isoformat(),
            "user_ids": model.user_ids,
            "item_ids": model.item_ids,
            "popularity": model.popularity,
            "activity": model.activity,
            "expertise_users": model.expertise_users,
            "graph_nodes": model.graph.nodes if model.graph else [],
            "als_config": dataclasses.asdict(model.als.config) if model.als else None,
        }
        (tmp / "meta.json").write_text(json.dumps(meta))
        if model.als is not None and model.als.user_factors is not None:
            np.savez_compressed(tmp / "als.npz", users=model.als.user_factors, items=model.als.item_factors)
        if model.expertise is not None:
            np.save(tmp / "expertise.npy", model.expertise)
        if model.graph is not None and model.graph.size:
            sparse.save_npz(tmp / "graph.npz", model.graph.adjacency)
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)

    def load(self, tenant_id: str) -> TenantModel | None:
        path = self._path(tenant_id)
        if not (path / "meta.json").exists():
            return None
        try:
            meta = json.loads((path / "meta.json").read_text())
            als = None
            if meta.get("als_config") and (path / "als.npz").exists():
                arrays = np.load(path / "als.npz")
                als = ImplicitALS(ALSConfig(**meta["als_config"]))
                als.user_factors = arrays["users"]
                als.item_factors = arrays["items"]
            graph = None
            if (path / "graph.npz").exists():
                nodes = meta["graph_nodes"]
                graph = WorkGraph(
                    nodes, {n: i for i, n in enumerate(nodes)}, sparse.load_npz(path / "graph.npz")
                )
            expertise = np.load(path / "expertise.npy") if (path / "expertise.npy").exists() else None
            return TenantModel(
                tenant_id=tenant_id,
                trained_at=datetime.fromisoformat(meta["trained_at"]),
                user_ids=meta["user_ids"],
                item_ids=meta["item_ids"],
                als=als,
                graph=graph,
                popularity=meta["popularity"],
                activity=meta["activity"],
                expertise_users=meta["expertise_users"],
                expertise=expertise,
            )
        except Exception:  # noqa: BLE001
            log.exception("model_load_failed", extra={"tenant": tenant_id})
            return None
