"""Composition root. Builds and wires every service from settings exactly once."""

from __future__ import annotations

from dataclasses import dataclass

from vantix.assist import AssistService
from vantix.config import LexicalBackend, Settings, VectorBackend
from vantix.embeddings import build_embedder
from vantix.embeddings.base import Embedder
from vantix.events.processor import EventProcessor
from vantix.index import build_lexical_index, build_vector_index
from vantix.index.base import LexicalIndex, VectorIndex
from vantix.ingestion import Chunker, IngestionQueue, IngestionService
from vantix.observability.logging import get_logger
from vantix.rag import RagService, build_llm
from vantix.rag.llm import LLM
from vantix.recommender import (
    ModelRegistry,
    RecommendationEngine,
    RecommenderTrainer,
    TrainingScheduler,
)
from vantix.retrieval import HybridRetriever, build_reranker
from vantix.retrieval.rerankers import Reranker
from vantix.storage import Repository

log = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    repository: Repository
    embedder: Embedder
    vector_index: VectorIndex
    lexical_index: LexicalIndex
    reranker: Reranker
    llm: LLM
    ingestion: IngestionService
    queue: IngestionQueue
    retriever: HybridRetriever
    registry: ModelRegistry
    trainer: RecommenderTrainer
    scheduler: TrainingScheduler
    recommender: RecommendationEngine
    rag: RagService
    assist: AssistService
    events: EventProcessor

    @classmethod
    def build(cls, settings: Settings) -> Container:
        repository = Repository(settings.database_url)
        repository.init_schema()
        embedder = build_embedder(settings)
        vector_index = build_vector_index(settings, embedder.dim)
        lexical_index = build_lexical_index(settings)
        reranker = build_reranker(settings)
        llm = build_llm(settings)
        chunker = Chunker(
            settings.chunk_target_tokens, settings.chunk_overlap_tokens, settings.chunk_min_tokens
        )
        ingestion = IngestionService(repository, embedder, vector_index, lexical_index, chunker)
        queue = IngestionQueue(
            ingestion, repository, settings.ingestion_workers, settings.ingestion_queue_size
        )
        retriever = HybridRetriever(settings, embedder, vector_index, lexical_index, reranker, repository)
        registry = ModelRegistry(settings.model_dir)
        trainer = RecommenderTrainer(settings, repository, embedder.dim)
        scheduler = TrainingScheduler(trainer, registry, repository, settings.rec_training_interval_seconds)
        recommender = RecommendationEngine(settings, repository, retriever, vector_index, embedder, registry)
        rag = RagService(settings, retriever, repository, llm, recommender)
        assist = AssistService(recommender, rag)
        events = EventProcessor(repository, queue, scheduler)
        queue.on_complete(scheduler.mark_dirty)
        return cls(
            settings,
            repository,
            embedder,
            vector_index,
            lexical_index,
            reranker,
            llm,
            ingestion,
            queue,
            retriever,
            registry,
            trainer,
            scheduler,
            recommender,
            rag,
            assist,
            events,
        )

    def start(self) -> None:
        """Warm in memory indexes from the system of record and start background workers."""
        if (
            self.settings.vector_backend is VectorBackend.MEMORY
            or self.settings.lexical_backend is LexicalBackend.MEMORY
        ):
            for tenant_id in self.repository.list_tenants():
                count = self.ingestion.rebuild_indexes(tenant_id)
                log.info("index_warmed", extra={"tenant": tenant_id, "chunks": count})
        self.queue.start()
        self.scheduler.start()

    def stop(self) -> None:
        self.scheduler.stop()
        self.queue.stop()
        self.assist.close()
        self.retriever.close()
        for closeable in (self.embedder, self.vector_index, self.lexical_index, self.reranker, self.llm):
            try:
                closeable.close()
            except Exception:  # noqa: BLE001
                log.exception("close_failed")
        self.repository.engine.dispose()
