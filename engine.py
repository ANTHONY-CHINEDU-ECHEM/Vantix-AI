"""Hybrid recommendation engine.

Recommendation is framed as multi source candidate generation followed by signal scoring,
calibrated blending, business rules, diversification and explanation:

    sources  ->  candidates  ->  signals  ->  blend  ->  rules  ->  MMR  ->  reasons

Signals:

* content        cosine between the user's real time interest profile and the item
* context        relevance to the current task, page or query (hybrid retrieval)
* collaborative  implicit ALS score from a user vector folded in at request time
* graph          Personalized PageRank over the user, item, project and tag graph
* popularity     time decayed engagement across the tenant
* urgency        priority and due date pressure (tasks only)
* evidence       related items a person authored or was assigned (experts only)
* activity       recent activity level of a person (experts only)

Signals that are unavailable for a request (a brand new user has no collaborative vector,
for example) are dropped and the remaining weights are renormalised, which handles cold
start gracefully instead of degrading every score.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np

from vantix.config import Settings
from vantix.domain.models import (
    INTERACTION_WEIGHTS,
    InteractionType,
    Item,
    ItemKind,
    Principal,
    Reason,
    Recommendation,
    TaskStatus,
)
from vantix.embeddings.base import Embedder, l2_normalise
from vantix.index.base import SearchFilter, VectorIndex
from vantix.observability.metrics import Timer
from vantix.recommender.graph import ITEM, PROJECT, USER, node_key
from vantix.recommender.model import ModelRegistry, TenantModel, decay
from vantix.retrieval.fusion import mmr
from vantix.retrieval.hybrid import HybridRetriever, RetrievalRequest
from vantix.security.acl import can_view_item
from vantix.storage.repository import Repository, StoredItem

_NEGATIVE = {InteractionType.DISMISS.value, InteractionType.DOWNVOTE.value}
_KNOWLEDGE_KINDS = [k for k in ItemKind if k.is_knowledge]


class RecommendationKind(StrEnum):
    DOCUMENTS = "documents"
    TASKS = "tasks"
    EXPERTS = "experts"
    SIMILAR = "similar"


@dataclass(slots=True)
class RecommendationContext:
    item_id: str | None = None
    query: str | None = None
    project_id: str | None = None


@dataclass(slots=True)
class RecommendationQuery:
    tenant_id: str
    principal: Principal
    kind: RecommendationKind
    k: int = 10
    context: RecommendationContext = field(default_factory=RecommendationContext)
    item_kinds: list[ItemKind] | None = None
    project_ids: list[str] | None = None
    exclude_ids: set[str] = field(default_factory=set)
    candidate_user_ids: list[str] | None = None
    include_assigned_to_others: bool = False
    include_self: bool = False
    diversify: bool = True


@dataclass(slots=True)
class _UserState:
    positives: dict[str, float]
    negatives: set[str]
    recent_titles: dict[str, str]
    profile: np.ndarray | None
    history_ids: list[str]
    history_matrix: np.ndarray | None
    user_factor: np.ndarray | None


class RecommendationEngine:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        retriever: HybridRetriever,
        vector_index: VectorIndex,
        embedder: Embedder,
        registry: ModelRegistry,
    ) -> None:
        self.settings = settings
        self.repo = repository
        self.retriever = retriever
        self.vectors = vector_index
        self.embedder = embedder
        self.registry = registry
        self.weights = {
            RecommendationKind.DOCUMENTS: Settings.parse_weights(settings.rec_weights_documents),
            RecommendationKind.TASKS: Settings.parse_weights(settings.rec_weights_tasks),
            RecommendationKind.SIMILAR: Settings.parse_weights(settings.rec_weights_similar),
            RecommendationKind.EXPERTS: Settings.parse_weights(settings.rec_weights_experts),
        }

    # Entry point ----------------------------------------------------------------------
    def recommend(self, q: RecommendationQuery) -> list[Recommendation]:
        with Timer(f"recommend_{q.kind.value}"):
            if q.kind is RecommendationKind.EXPERTS:
                return self._recommend_experts(q)
            return self._recommend_items(q)

    # Shared state ---------------------------------------------------------------------
    def user_state(self, tenant_id: str, principal: Principal, model: TenantModel) -> _UserState:
        s = self.settings
        now = datetime.now(UTC)
        history = self.repo.user_interactions(
            tenant_id, principal.user_id, limit=s.rec_profile_max_interactions
        )
        positives: dict[str, float] = defaultdict(float)
        negatives: set[str] = set()
        latest_type: dict[str, str] = {}
        for item_id, itype, weight, occurred_at in history:  # newest first
            latest_type.setdefault(item_id, itype)
            age = (now - occurred_at).total_seconds() / 86400.0
            positives[item_id] += weight * decay(age, s.rec_interaction_half_life_days)
        for item_id, itype in latest_type.items():
            if itype in _NEGATIVE or positives[item_id] <= 0:
                negatives.add(item_id)
        positives = {i: w for i, w in positives.items() if w > 0 and i not in negatives}

        stored = self.repo.get_items(tenant_id, positives) if positives else {}
        visible = {i: st for i, st in stored.items() if can_view_item(principal, st.item)}
        history_ids = [i for i in visible if visible[i].embedding is not None]
        profile = None
        history_matrix = None
        if history_ids:
            history_matrix = np.vstack([visible[i].embedding for i in history_ids])  # type: ignore[misc]
            w = np.asarray([np.log1p(positives[i]) for i in history_ids], dtype=np.float32)
            profile = l2_normalise((history_matrix * w[:, None]).sum(axis=0))

        user_factor = None
        if model.als is not None:
            known = [(model.item_index[i], positives[i]) for i in positives if i in model.item_index]
            if known:
                idx = np.asarray([k for k, _ in known], dtype=np.int64)
                strength = np.asarray([v for _, v in known], dtype=np.float32)
                user_factor = model.als.fold_in(idx, strength)

        return _UserState(
            positives=positives,
            negatives=negatives,
            recent_titles={i: st.item.title for i, st in visible.items()},
            profile=profile,
            history_ids=history_ids,
            history_matrix=history_matrix,
            user_factor=user_factor,
        )

    def affinity(self, tenant_id: str, principal: Principal) -> dict[str, float]:
        """Item affinity for search personalisation: graph proximity from the user node."""
        model = self.registry.get(tenant_id)
        if model.graph is None:
            return {}
        ppr = model.graph.personalized_pagerank(
            {node_key(USER, principal.user_id): 1.0}, damping=self.settings.rec_ppr_damping
        )
        scores = {k[2:]: v for k, v in ppr.items() if k.startswith(ITEM + ":")}
        if not scores:
            return {}
        top = max(scores.values())
        return {k: v / top for k, v in scores.items()}

    # Item recommendations -------------------------------------------------------------
    def _recommend_items(self, q: RecommendationQuery) -> list[Recommendation]:
        s = self.settings
        pool = s.rec_candidate_pool
        model = self.registry.get(q.tenant_id)
        state = self.user_state(q.tenant_id, q.principal, model)

        kinds = self._target_kinds(q)
        flt = SearchFilter(
            principal=q.principal,
            kinds=kinds,
            project_ids=q.project_ids,
            exclude_item_ids=set(q.exclude_ids) | state.negatives,
            active_tasks_only=q.kind is RecommendationKind.TASKS,
        )

        context_item = self._context_item(q)
        context_text = self._context_text(q, context_item)
        if context_item is not None:
            flt.exclude_item_ids.add(context_item.item.id)

        candidates: dict[str, set[str]] = defaultdict(set)
        context_scores: dict[str, float] = {}
        if context_text:
            hits = self.retriever.retrieve(
                RetrievalRequest(q.tenant_id, context_text, flt, k=min(pool, 100), diversify=False)
            )
            for hit in hits:
                item_id = hit.chunk.item_id
                context_scores[item_id] = max(context_scores.get(item_id, 0.0), hit.score)
                candidates[item_id].add("context")

        if state.profile is not None and q.kind is not RecommendationKind.SIMILAR:
            for hit in self.vectors.search(q.tenant_id, state.profile, pool, flt):
                candidates[hit.chunk.item_id].add("content")

        collab_scores: dict[str, float] = {}
        if model.als is not None:
            if (
                q.kind is RecommendationKind.SIMILAR
                and context_item
                and context_item.item.id in model.item_index
            ):
                for idx, score in model.als.similar_items(model.item_index[context_item.item.id], pool):
                    collab_scores[model.item_ids[idx]] = score
            elif state.user_factor is not None:
                raw = model.als.score_items(state.user_factor)
                top = np.argsort(-raw)[:pool]
                collab_scores = {model.item_ids[int(i)]: float(raw[int(i)]) for i in top}
            for item_id in collab_scores:
                candidates[item_id].add("collaborative")

        graph_scores = self._graph_item_scores(q, model, context_item)
        for item_id in sorted(graph_scores, key=graph_scores.__getitem__, reverse=True)[:pool]:
            candidates[item_id].add("graph")

        if q.kind is not RecommendationKind.SIMILAR:
            for item_id in sorted(model.popularity, key=model.popularity.__getitem__, reverse=True)[:pool]:
                candidates[item_id].add("popularity")

        if q.kind is RecommendationKind.TASKS and len(candidates) < q.k * 3:
            # Cold start fallback: open tasks in the user's projects.
            for st in self.repo.iter_items(q.tenant_id, kinds=[ItemKind.TASK], active_tasks_only=True):
                candidates[st.item.id].add("fallback")
                if len(candidates) >= pool:
                    break

        stored = self.repo.get_items(q.tenant_id, candidates)
        eligible = {i: st for i, st in stored.items() if self._eligible(q, st, flt, state)}
        if not eligible:
            return []

        ids = list(eligible)
        emb = self._embedding_matrix(ids, eligible)
        signals: dict[str, dict[str, float]] = {}

        if state.profile is not None and q.kind is not RecommendationKind.SIMILAR:
            signals["content"] = _cosines(ids, emb, state.profile)

        context_vec = self._context_vector(context_item, q.context.query)
        if context_scores or context_vec is not None:
            cos = _cosines(ids, emb, context_vec) if context_vec is not None else {}
            signals["context"] = {i: max(context_scores.get(i, 0.0), cos.get(i, 0.0)) for i in ids}

        if collab_scores or (state.user_factor is not None and model.als is not None):
            raw = {}
            for i in ids:
                if i in collab_scores:
                    raw[i] = collab_scores[i]
                elif state.user_factor is not None and model.als is not None and i in model.item_index:
                    raw[i] = float(model.als.item_factors[model.item_index[i]] @ state.user_factor)  # type: ignore[index]
            if raw:
                signals["collaborative"] = _minmax_fill(raw, ids)

        if graph_scores:
            signals["graph"] = _scale_by_max({i: graph_scores.get(i, 0.0) for i in ids})

        if model.popularity and q.kind is not RecommendationKind.SIMILAR:
            signals["popularity"] = {i: model.popularity.get(i, 0.0) for i in ids}

        if q.kind is RecommendationKind.TASKS:
            signals["urgency"] = {i: _urgency(eligible[i].item) for i in ids}

        weights = {k: w for k, w in self.weights[q.kind].items() if k in signals and w > 0}
        if not weights:  # nothing personal or contextual at all: fall back to recency
            signals["recency"] = _recency({i: eligible[i].item for i in ids})
            weights = {"recency": 1.0}
        total = sum(weights.values())
        weights = {k: w / total for k, w in weights.items()}

        scored: list[tuple[str, float]] = []
        for i in ids:
            score = sum(w * signals[name].get(i, 0.0) for name, w in weights.items())
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        if q.diversify and len(scored) > q.k:
            head = scored[: q.k * 4]
            pos = {i: n for n, i in enumerate(ids)}
            score_of = dict(head)
            selected = mmr(
                [i for i, _ in head],
                relevance=lambda i: score_of[i],
                similarity=lambda a, b: float(max(0.0, emb[pos[a]] @ emb[pos[b]])),
                k=q.k,
                lambda_=s.rec_mmr_lambda,
            )
            scored = [(i, score_of[i]) for i in selected]
        else:
            scored = scored[: q.k]

        recs: list[Recommendation] = []
        pos_all = {i: n for n, i in enumerate(ids)}
        for item_id, score in scored:
            item = eligible[item_id].item
            item_signals = {name: round(signals[name].get(item_id, 0.0), 6) for name in weights}
            reasons = self._item_reasons(
                q, item, item_signals, weights, state, context_item, emb[pos_all[item_id]], model
            )
            recs.append(
                Recommendation(
                    target_id=item_id,
                    target_type=item.kind.value,
                    score=round(float(score), 6),
                    title=item.title,
                    url=item.url,
                    reasons=reasons,
                    signals=item_signals,
                )
            )
        return recs

    def _target_kinds(self, q: RecommendationQuery) -> list[ItemKind] | None:
        if q.item_kinds:
            return q.item_kinds
        if q.kind is RecommendationKind.TASKS:
            return [ItemKind.TASK]
        if q.kind is RecommendationKind.DOCUMENTS:
            return list(_KNOWLEDGE_KINDS)
        return None

    def _eligible(self, q: RecommendationQuery, st: StoredItem, flt: SearchFilter, state: _UserState) -> bool:
        item = st.item
        if not can_view_item(q.principal, item):
            return False
        if flt.kinds and item.kind not in flt.kinds:
            return False
        if item.id in flt.exclude_item_ids:
            return False
        if q.project_ids and item.project_id not in q.project_ids:
            return False
        if q.kind is RecommendationKind.TASKS:
            if item.status is not None and not item.status.is_active:
                return False
            assigned = item.assignee_ids
            if assigned and q.principal.user_id not in assigned and not q.include_assigned_to_others:
                return False
            if item.project_id and q.principal.project_ids and item.project_id not in q.principal.project_ids:
                return False
        has_context = bool(q.context.item_id or q.context.query)
        if q.kind is RecommendationKind.DOCUMENTS and not has_context and item.id in state.positives:
            # Discovery feeds should not resurface documents the user engaged with recently.
            # Contextual panels (a task or a query) always show the most relevant documents.
            return state.positives[item.id] < INTERACTION_WEIGHTS[InteractionType.VIEW] * 0.5
        return True

    def _context_item(self, q: RecommendationQuery) -> StoredItem | None:
        if not q.context.item_id:
            return None
        stored = self.repo.get_item(q.tenant_id, q.context.item_id)
        if stored is None or not can_view_item(q.principal, stored.item):
            return None
        return stored

    @staticmethod
    def _context_text(q: RecommendationQuery, context_item: StoredItem | None) -> str | None:
        parts: list[str] = []
        if q.context.query:
            parts.append(q.context.query)
        if context_item is not None:
            item = context_item.item
            parts.append(f"{item.title}\n{item.body[:1200]}")
            if item.tags:
                parts.append(" ".join(item.tags))
        text = "\n".join(p for p in parts if p).strip()
        return text or None

    def _context_vector(self, context_item: StoredItem | None, query: str | None) -> np.ndarray | None:
        vectors = []
        if context_item is not None and context_item.embedding is not None:
            vectors.append(context_item.embedding)
        if query:
            vectors.append(self.embedder.embed_query(query))
        if not vectors:
            return None
        return l2_normalise(np.mean(np.vstack(vectors), axis=0))

    def _graph_item_scores(
        self, q: RecommendationQuery, model: TenantModel, context_item: StoredItem | None
    ) -> dict[str, float]:
        if model.graph is None or model.graph.size == 0:
            return {}
        seeds: dict[str, float] = {}
        if q.kind is not RecommendationKind.SIMILAR:
            seeds[node_key(USER, q.principal.user_id)] = 1.0
        if context_item is not None:
            seeds[node_key(ITEM, context_item.item.id)] = 2.0
        project = q.context.project_id or (context_item.item.project_id if context_item else None)
        if project:
            seeds[node_key(PROJECT, project)] = 0.5
        ppr = model.graph.personalized_pagerank(seeds, damping=self.settings.rec_ppr_damping)
        prefix = ITEM + ":"
        return {k[len(prefix) :]: v for k, v in ppr.items() if k.startswith(prefix)}

    def _embedding_matrix(self, ids: list[str], stored: dict[str, StoredItem]) -> np.ndarray:
        dim = self.embedder.dim
        out = np.zeros((len(ids), dim), dtype=np.float32)
        for n, i in enumerate(ids):
            vec = stored[i].embedding
            if vec is not None and vec.shape[0] == dim:
                out[n] = vec
        return out

    # Explanations ---------------------------------------------------------------------
    def _item_reasons(
        self,
        q: RecommendationQuery,
        item: Item,
        signals: dict[str, float],
        weights: dict[str, float],
        state: _UserState,
        context_item: StoredItem | None,
        item_vec: np.ndarray,
        model: TenantModel,
    ) -> list[Reason]:
        contributions = sorted(
            ((name, weights[name] * signals.get(name, 0.0)) for name in weights), key=lambda x: -x[1]
        )
        reasons: list[Reason] = []
        for name, contribution in contributions:
            if contribution < 0.02 or len(reasons) >= 3:
                continue
            message = self._reason_message(name, q, item, state, context_item, item_vec, model)
            if message:
                reasons.append(Reason(code=name, message=message, weight=round(contribution, 4)))
        return reasons

    def _reason_message(
        self,
        name: str,
        q: RecommendationQuery,
        item: Item,
        state: _UserState,
        context_item: StoredItem | None,
        item_vec: np.ndarray,
        model: TenantModel,
    ) -> str | None:
        if name == "content":
            if state.history_matrix is not None and state.history_ids:
                sims = state.history_matrix @ item_vec
                best = state.history_ids[int(np.argmax(sims))]
                if best != item.id:
                    return f'Similar to "{state.recent_titles.get(best, best)}", which you worked on recently'
            return "Matches the topics you work on most"
        if name == "context":
            if context_item is not None:
                return f'Relevant to "{context_item.item.title}"'
            if q.context.query:
                return f'Matches "{q.context.query}"'
            return None
        if name == "collaborative":
            return "Colleagues with similar work patterns found this useful"
        if name == "graph":
            if context_item is not None and (
                item.id in context_item.item.linked_item_ids
                or context_item.item.id in item.linked_item_ids
                or item.parent_id == context_item.item.id
                or context_item.item.parent_id == item.id
            ):
                return f'Directly linked to "{context_item.item.title}"'
            if item.project_id and item.project_id in q.principal.project_ids:
                return f"Closely connected to your work in project {item.project_id}"
            return "Closely connected to items you work with"
        if name == "popularity":
            return "Trending in your workspace"
        if name == "urgency":
            return _urgency_message(item)
        if name == "recency":
            return "Recently updated"
        return None

    # Experts --------------------------------------------------------------------------
    def _recommend_experts(self, q: RecommendationQuery) -> list[Recommendation]:
        s = self.settings
        model = self.registry.get(q.tenant_id)
        context_item = self._context_item(q)
        context_text = self._context_text(q, context_item)
        context_vec = self._context_vector(context_item, q.context.query)
        if context_vec is None and context_text:
            context_vec = self.embedder.embed_query(context_text)

        signals: dict[str, dict[str, float]] = defaultdict(dict)
        evidence_items: dict[str, list[str]] = defaultdict(list)
        titles: dict[str, str] = {}

        if model.expertise is not None and context_vec is not None:
            sims = model.expertise @ context_vec
            for idx in np.argsort(-sims)[: s.rec_candidate_pool]:
                signals["content"][model.expertise_users[int(idx)]] = max(0.0, float(sims[int(idx)]))

        if model.graph is not None and model.graph.size:
            seeds: dict[str, float] = {}
            if context_item is not None:
                seeds[node_key(ITEM, context_item.item.id)] = 1.0
            project = q.context.project_id or (context_item.item.project_id if context_item else None)
            if project:
                seeds[node_key(PROJECT, project)] = 0.5
            if seeds:
                ppr = model.graph.personalized_pagerank(seeds, damping=s.rec_ppr_damping)
                users = {k[2:]: v for k, v in ppr.items() if k.startswith(USER + ":")}
                signals["graph"] = _scale_by_max(users)

        if context_text:
            flt = SearchFilter(principal=q.principal)
            if context_item is not None:
                flt.exclude_item_ids.add(context_item.item.id)
            hits = self.retriever.retrieve(
                RetrievalRequest(q.tenant_id, context_text, flt, k=30, diversify=False)
            )
            related = self.repo.get_items(q.tenant_id, {h.chunk.item_id for h in hits})
            relevance = {h.chunk.item_id: h.score for h in hits}
            for item_id, st in related.items():
                titles[item_id] = st.item.title
                people = set(st.item.assignee_ids)
                if st.item.author_id:
                    people.add(st.item.author_id)
                for person in people:
                    signals["evidence"][person] = signals["evidence"].get(person, 0.0) + relevance.get(
                        item_id, 0.0
                    )
                    evidence_items[person].append(item_id)
            signals["evidence"] = _scale_by_max(signals["evidence"])

        candidates = set().union(*[set(v) for v in signals.values()]) if signals else set()
        if q.candidate_user_ids is not None:
            candidates &= set(q.candidate_user_ids)
        if not q.include_self:
            candidates.discard(q.principal.user_id)
        if not candidates:
            return []
        if model.activity:
            signals["activity"] = {u: model.activity.get(u, 0.0) for u in candidates}

        weights = {
            k: w for k, w in self.weights[RecommendationKind.EXPERTS].items() if signals.get(k) and w > 0
        }
        total = sum(weights.values()) or 1.0
        weights = {k: w / total for k, w in weights.items()}
        scored = sorted(
            ((u, sum(w * signals[n].get(u, 0.0) for n, w in weights.items())) for u in candidates),
            key=lambda x: -x[1],
        )[: q.k]

        recs = []
        for user_id, score in scored:
            sig = {n: round(signals[n].get(user_id, 0.0), 6) for n in weights}
            reasons: list[Reason] = []
            items = evidence_items.get(user_id, [])
            if items:
                example = titles.get(items[0], items[0])
                count = len(items)
                noun = "item" if count == 1 else "items"
                reasons.append(
                    Reason(
                        code="evidence",
                        message=f'Worked on {count} closely related {noun}, including "{example}"',
                        weight=round(weights.get("evidence", 0.0) * sig.get("evidence", 0.0), 4),
                    )
                )
            if sig.get("content", 0.0) > 0.3:
                reasons.append(
                    Reason(
                        code="content",
                        message="Track record matches this topic",
                        weight=round(weights.get("content", 0.0) * sig["content"], 4),
                    )
                )
            if sig.get("graph", 0.0) > 0.2:
                reasons.append(
                    Reason(
                        code="graph",
                        message="Closely connected to this work and project",
                        weight=round(weights.get("graph", 0.0) * sig["graph"], 4),
                    )
                )
            if sig.get("activity", 0.0) > 0.5:
                reasons.append(
                    Reason(
                        code="activity",
                        message="Recently active",
                        weight=round(weights.get("activity", 0.0) * sig["activity"], 4),
                    )
                )
            recs.append(
                Recommendation(
                    target_id=user_id,
                    target_type="user",
                    score=round(float(score), 6),
                    reasons=reasons,
                    signals=sig,
                )
            )
        return recs


# Helpers ------------------------------------------------------------------------------
def _cosines(ids: list[str], matrix: np.ndarray, vector: np.ndarray) -> dict[str, float]:
    sims = matrix @ vector
    return {i: max(0.0, float(sims[n])) for n, i in enumerate(ids)}


def _scale_by_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / top for k, v in scores.items()}


def _minmax_fill(raw: dict[str, float], ids: list[str]) -> dict[str, float]:
    lo, hi = min(raw.values()), max(raw.values())
    span = hi - lo
    out = {}
    for i in ids:
        if i not in raw:
            out[i] = 0.0
        else:
            out[i] = 1.0 if span < 1e-12 else (raw[i] - lo) / span
    return out


def _urgency(item: Item) -> float:
    priority = (item.priority or 0) / 4.0
    due = 0.0
    if item.due_at is not None:
        days = (item.due_at - datetime.now(UTC)).total_seconds() / 86400.0
        due = 1.0 if days <= 0 else float(np.exp(-days / 7.0))
    blocked = 0.5 if item.status is TaskStatus.BLOCKED else 1.0
    return blocked * (0.5 * priority + 0.5 * due)


def _urgency_message(item: Item) -> str | None:
    if item.due_at is not None:
        days = (item.due_at - datetime.now(UTC)).total_seconds() / 86400.0
        if days < 0:
            return "Overdue"
        if days < 1:
            return "Due today"
        if days < 14:
            return f"Due in {int(round(days))} days"
    if (item.priority or 0) >= 3:
        return "High priority"
    return None


def _recency(items: dict[str, Item]) -> dict[str, float]:
    now = datetime.now(UTC)
    return {i: decay((now - it.updated_at).total_seconds() / 86400.0, 14.0) for i, it in items.items()}
