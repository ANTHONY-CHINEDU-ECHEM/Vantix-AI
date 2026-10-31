"""Central configuration for Vantix AI.

Every setting is read from environment variables prefixed with ``VANTIX_`` (or from a
``.env`` file in the working directory). Settings are grouped by subsystem so that an
integrating team can reason about one concern at a time.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorBackend(StrEnum):
    MEMORY = "memory"
    QDRANT = "qdrant"


class LexicalBackend(StrEnum):
    MEMORY = "memory"
    OPENSEARCH = "opensearch"


class EmbeddingProvider(StrEnum):
    HASHING = "hashing"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI_COMPATIBLE = "openai_compatible"


class RerankerProvider(StrEnum):
    NONE = "none"
    LEXICAL = "lexical"
    CROSS_ENCODER = "cross_encoder"
    HTTP = "http"


class LLMProvider(StrEnum):
    EXTRACTIVE = "extractive"
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Runtime settings. Defaults give a fully offline, self contained development stack."""

    model_config = SettingsConfigDict(
        env_prefix="VANTIX_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Service
    environment: str = "development"
    # containers must bind all interfaces
    host: str = "0.0.0.0"  # nosec B104
    port: int = 8080
    workers: int = 1
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: list[str] = Field(default_factory=list)

    # Security. Format: "tenant_id:api_key,tenant_id:api_key"
    api_keys: str = "demo:demo_key"
    admin_api_keys: str = ""
    webhook_secret: str = ""
    webhook_tolerance_seconds: int = 300
    rate_limit_per_minute: int = 1200

    # Storage
    database_url: str = "sqlite:///./data/vantix.db"
    model_dir: Path = Path("./data/models")

    # Embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HASHING
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""
    embedding_batch_size: int = 64
    embedding_timeout_seconds: float = 30.0

    # Vector index
    vector_backend: VectorBackend = VectorBackend.MEMORY
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "vantix_chunks"
    qdrant_hnsw_ef: int = 128

    # Lexical index
    lexical_backend: LexicalBackend = LexicalBackend.MEMORY
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "vantix_chunks"
    opensearch_username: str = ""
    opensearch_password: str = ""

    # Chunking
    chunk_target_tokens: int = 350
    chunk_overlap_tokens: int = 50
    chunk_min_tokens: int = 40

    # Retrieval
    retrieval_dense_k: int = 100
    retrieval_sparse_k: int = 100
    retrieval_fusion_k: int = 60
    retrieval_dense_weight: float = 1.0
    retrieval_sparse_weight: float = 1.0
    retrieval_personal_weight: float = 0.15
    retrieval_mmr_lambda: float = 0.75
    retrieval_max_chunks_per_document: int = 3
    retrieval_cache_ttl_seconds: int = 120

    # Reranking
    reranker_provider: RerankerProvider = RerankerProvider.LEXICAL
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_url: str = ""
    reranker_api_key: str = ""
    reranker_top_n: int = 50

    # Generation
    llm_provider: LLMProvider = LLMProvider.EXTRACTIVE
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 700
    llm_timeout_seconds: float = 60.0
    rag_context_token_budget: int = 3000
    # Abstain unless a passage clears at least one of these relevance gates.
    # Calibrate rag_min_dense_similarity per embedding model with the evaluation harness.
    rag_min_dense_similarity: float = 0.2
    rag_min_lexical_coverage: float = 0.3

    # Recommender
    rec_als_factors: int = 64
    rec_als_regularization: float = 0.05
    rec_als_iterations: int = 15
    rec_als_alpha: float = 20.0
    rec_ppr_damping: float = 0.85
    rec_ppr_iterations: int = 30
    rec_interaction_half_life_days: float = 30.0
    rec_popularity_half_life_days: float = 7.0
    rec_profile_max_interactions: int = 200
    rec_candidate_pool: int = 300
    rec_training_interval_seconds: int = 3600
    rec_weights_documents: str = "content:0.30,collaborative:0.25,graph:0.20,popularity:0.10,context:0.15"
    rec_weights_tasks: str = (
        "content:0.25,collaborative:0.15,graph:0.20,popularity:0.05,context:0.25,urgency:0.10"
    )
    rec_weights_similar: str = "context:0.50,collaborative:0.25,graph:0.25"
    rec_weights_experts: str = "content:0.35,graph:0.30,evidence:0.25,activity:0.10"
    rec_mmr_lambda: float = 0.8

    # Workers
    # liveness only
    worker_heartbeat_file: Path = Path("/tmp/vantix_worker_heartbeat")  # nosec B108
    model_refresh_seconds: float = 30.0
    ingestion_workers: int = 2
    ingestion_queue_size: int = 10000

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def tenant_keys(self) -> dict[str, str]:
        """Return a mapping of api_key to tenant_id."""
        return _parse_pairs(self.api_keys)

    def admin_keys(self) -> dict[str, str]:
        return _parse_pairs(self.admin_api_keys)

    @staticmethod
    def parse_weights(spec: str) -> dict[str, float]:
        weights: dict[str, float] = {}
        for part in spec.split(","):
            if not part.strip():
                continue
            name, _, value = part.partition(":")
            weights[name.strip()] = float(value)
        return weights


def _parse_pairs(spec: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        tenant, sep, key = part.partition(":")
        if not sep or not tenant or not key:
            raise ValueError(f"Invalid key specification '{part}'. Expected tenant_id:api_key")
        mapping[key.strip()] = tenant.strip()
    return mapping


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
