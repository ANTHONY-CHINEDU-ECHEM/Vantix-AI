# Vantix AI

**Hybrid recommender and retrieval augmented generation engine for project management platforms.**

Vantix AI is the intelligence layer built by Phexara Ltd to sit behind TaskTown and any other Phexara product that needs to connect people with the right knowledge, the right work and the right colleagues. It combines a production grade recommender system with a retrieval augmented generation (RAG) pipeline over a very large knowledge base, and exposes both through a single, versioned HTTP API that a host platform can integrate in days rather than months.

Vantix answers five questions for every user, in real time and strictly within that user's permissions:

1. **What do I need to know?** Hybrid search across tasks, documents, wiki pages, comments, files, messages and decisions.
2. **What is the answer?** Grounded answers with numbered citations, and an explicit abstention when the knowledge base does not contain the answer.
3. **What should I read next?** Personalised knowledge recommendations that learn from behaviour.
4. **What should I work on next?** Ranked task recommendations that weigh relevance, collaboration patterns, priority and due dates.
5. **Who can help?** Expert recommendations based on authorship, assignment history, topical expertise and the work graph.

Every recommendation carries machine readable signals and human readable reasons, so the host interface can always explain why something was suggested.

## Capabilities at a glance

* **Hybrid retrieval.** Dense vector recall and Okapi BM25 lexical recall run in parallel, are merged with weighted reciprocal rank fusion, personalised with graph affinity, reranked by a second stage model, capped per item and diversified with maximal marginal relevance.
* **Hybrid recommendation.** Content based profiles, implicit feedback matrix factorisation (ALS) with real time fold in, Personalized PageRank over a heterogeneous work graph, time decayed popularity, contextual relevance and task urgency, blended with calibrated weights that renormalise automatically for cold start users and items.
* **Grounded generation.** Token budgeted context assembly, numbered sources, citation validation, prompt injection hardening, streaming output and pluggable language models (OpenAI compatible endpoints, Anthropic, or a fully offline extractive answerer).
* **Contextual assist.** One call returns a grounded answer, relevant documents, similar tasks and suggested experts for the task a user is looking at, computed in parallel.
* **Built for very large knowledge bases.** Qdrant with int8 scalar quantisation, on disk vectors and tenant aware payload indexes; OpenSearch for distributed BM25; PostgreSQL as the system of record; change aware ingestion that never re embeds unchanged content.
* **Enterprise controls.** Multitenancy with hard isolation, document level access control enforced inside the index query and again in memory, signed and idempotent webhooks, rate limiting, structured logs, Prometheus metrics, readiness probes and an offline evaluation harness.
* **Runs anywhere.** The default configuration is completely offline and dependency light, so the whole system runs on a laptop and in CI with no model downloads and no external services.

## System architecture

The diagram below shows every runtime component and how data moves between the host platform, the Vantix service, the data plane and the model providers.

```mermaid
flowchart TB
    subgraph HOST["HOST PLATFORM: ProWin and other Phexara products"]
        direction LR
        UI["Web and mobile clients"]
        BFF["Host backend<br/>identity, projects, permissions"]
        OUTBOX[("Transactional outbox")]
        RELAY["Outbox relay<br/>batched, signed, retried"]
    end

    subgraph API["VANTIX API LAYER: stateless FastAPI replicas"]
        direction LR
        AUTHN["Bearer API key<br/>resolves tenant"]
        LIMIT["Token bucket<br/>rate limiter"]
        SIGN["Webhook verifier<br/>HMAC SHA256 and replay window"]
        ROUTES["Versioned routes /v1<br/>search, ask, recommendations,<br/>assist, events, feedback, admin"]
    end

    subgraph WRITE["WRITE PATH"]
        direction TB
        EVENTS["Event processor<br/>idempotent on event id"]
        QUEUE["Ingestion queue<br/>bounded, persisted job status"]
        PIPELINE["Ingestion pipeline<br/>normalise, chunk, fingerprint"]
        EMBED["Embedding service<br/>batched calls, query cache"]
    end

    subgraph READ["READ PATH"]
        direction TB
        ASSIST["Contextual assist<br/>parallel orchestration"]
        RAG["RAG service<br/>abstention, context packing,<br/>citation validation"]
        RECO["Recommendation engine<br/>candidates, signals, blend,<br/>rules, MMR, reasons"]
        RETRIEVER["Hybrid retriever"]
        DENSE["Dense recall"]
        SPARSE["Lexical recall BM25"]
        FUSION["Weighted RRF fusion<br/>and personal affinity"]
        RERANK["Second stage reranker"]
        DIVERSE["Per item cap and MMR"]
    end

    subgraph LEARN["LEARNING LOOP"]
        direction TB
        SCHED["Training scheduler<br/>dirty tenant tracking"]
        TRAINER["Recommender trainer"]
        ALS["Implicit ALS<br/>user and item factors"]
        GRAPH["Work graph<br/>Personalized PageRank"]
        STATS["Popularity, activity<br/>and expertise vectors"]
        REGISTRY["Model registry<br/>atomic swap, warm restart"]
    end

    subgraph DATA["DATA PLANE"]
        direction LR
        PG[("PostgreSQL<br/>system of record")]
        QDRANT[("Qdrant<br/>vector index")]
        OPENSEARCH[("OpenSearch<br/>lexical index")]
        STORE[("Model store")]
    end

    subgraph PROVIDERS["MODEL PROVIDERS: pluggable"]
        direction LR
        EMBMODEL["Embedding model"]
        RERANKMODEL["Reranker model"]
        LLM["Large language model"]
    end

    subgraph OPS["OPERATIONS"]
        direction LR
        METRICS["Prometheus metrics"]
        LOGS["Structured JSON logs<br/>request correlation"]
        PROBES["Health and readiness"]
    end

    UI ==> BFF
    BFF ==> AUTHN
    BFF ==> OUTBOX
    OUTBOX ==> RELAY
    RELAY ==> SIGN
    AUTHN ==> LIMIT
    LIMIT ==> ROUTES
    SIGN ==> ROUTES

    ROUTES ==> EVENTS
    EVENTS ==> PG
    EVENTS ==> QUEUE
    QUEUE ==> PIPELINE
    PIPELINE ==> EMBED
    EMBED ==> EMBMODEL
    PIPELINE ==> PG
    PIPELINE ==> QDRANT
    PIPELINE ==> OPENSEARCH

    ROUTES ==> ASSIST
    ROUTES ==> RAG
    ROUTES ==> RECO
    ROUTES ==> RETRIEVER
    ASSIST ==> RAG
    ASSIST ==> RECO
    RAG ==> RETRIEVER
    RAG ==> LLM
    RECO ==> RETRIEVER
    RECO ==> PG
    RECO ==> REGISTRY
    RETRIEVER ==> DENSE
    RETRIEVER ==> SPARSE
    DENSE ==> QDRANT
    SPARSE ==> OPENSEARCH
    DENSE ==> FUSION
    SPARSE ==> FUSION
    FUSION ==> RERANK
    RERANK ==> RERANKMODEL
    RERANK ==> DIVERSE

    EVENTS ==> SCHED
    SCHED ==> TRAINER
    TRAINER ==> PG
    TRAINER ==> ALS
    TRAINER ==> GRAPH
    TRAINER ==> STATS
    ALS ==> REGISTRY
    GRAPH ==> REGISTRY
    STATS ==> REGISTRY
    REGISTRY ==> STORE

    ROUTES ==> METRICS
    ROUTES ==> LOGS
    ROUTES ==> PROBES

    classDef default fill:none
    style HOST fill:none
    style API fill:none
    style WRITE fill:none
    style READ fill:none
    style LEARN fill:none
    style DATA fill:none
    style PROVIDERS fill:none
    style OPS fill:none
```

### Architectural principles

* **The host owns identity, Vantix owns ranking.** Vantix never decides who belongs to a project or a team. The host platform sends a principal (user, projects, groups, admin flag) with every read request, and Vantix evaluates item level access control against it. This keeps a single source of truth for permissions and avoids synchronising membership.
* **One system of record, many derived views.** PostgreSQL holds every item, chunk, embedding, interaction and job. Qdrant, OpenSearch and the trained models are derived views that can be rebuilt at any time with `vantix reindex` and `vantix train`.
* **Asynchronous writes, synchronous reads.** Content changes flow through an idempotent event pipeline and a bounded background queue. Reads are synchronous, cached per tenant and invalidated precisely by a tenant content version.
* **Defence in depth for access control.** Permission filters are executed inside the vector and lexical index queries, and every returned chunk is checked again in memory before it can reach a response, a prompt or an explanation.
* **Graceful degradation.** Every signal is optional. A new user without history still gets contextual, graph and popularity based results; a new item without interactions still gets content and contextual scores; an unavailable language model can be replaced by the offline extractive answerer.
* **Pluggable providers.** Embeddings, vector index, lexical index, reranker and language model are each behind a small abstract interface and selected by configuration.

### Ingestion and learning lifecycle

```mermaid
flowchart LR
    CHANGE["Change in the host<br/>task, page, comment, file"] ==> OUTBOX[("Outbox row")]
    OUTBOX ==> WEBHOOK["POST /v1/events<br/>signed batch"]
    WEBHOOK ==> VERIFY{"Signature valid and<br/>timestamp in window?"}
    VERIFY ==> REJECT["If invalid:<br/>401 rejected"]
    VERIFY ==> DEDUPE{"If valid: event id<br/>already processed?"}
    DEDUPE ==> SKIP["If processed:<br/>counted as duplicate"]
    DEDUPE ==> INTERACTIONS[("New interaction:<br/>interactions table")]
    DEDUPE ==> JOB["New item upsert or delete:<br/>ingestion job queued with id"]
    JOB ==> FINGERPRINT{"Content and metadata<br/>fingerprints"}
    FINGERPRINT ==> NOOP["Both unchanged:<br/>skipped"]
    FINGERPRINT ==> REFRESH["Metadata only changed:<br/>refresh index payload,<br/>reuse stored vectors"]
    FINGERPRINT ==> CHUNK["Content changed:<br/>normalise and chunk,<br/>heading aware, overlapping"]
    CHUNK ==> VECTORS["Batch embed chunks<br/>and item vector"]
    VECTORS ==> PERSIST[("PostgreSQL<br/>transactional upsert")]
    PERSIST ==> INDEXES[("Qdrant and OpenSearch<br/>upsert and stale chunk removal")]
    REFRESH ==> INDEXES
    INDEXES ==> VERSION["Tenant content version<br/>incremented"]
    VERSION ==> CACHE["Retrieval cache<br/>invalidated"]
    INTERACTIONS ==> DIRTY["Tenant marked dirty"]
    JOB ==> DIRTY
    DIRTY ==> RETRAIN["Scheduled retraining<br/>ALS, graph, popularity, expertise"]
    RETRAIN ==> SWAP["Model registry<br/>atomic swap and persist"]

    classDef default fill:none
```

### Retrieval and recommendation pipeline

```mermaid
flowchart TB
    subgraph RETRIEVAL["HYBRID RETRIEVAL"]
        direction LR
        Q["Query and principal"] ==> DR["Dense recall<br/>top 100 chunks"]
        Q ==> LR2["BM25 recall<br/>top 100 chunks"]
        DR ==> RRF["Weighted RRF<br/>k equals 60"]
        LR2 ==> RRF
        RRF ==> AFF["Graph affinity boost<br/>for the user"]
        AFF ==> RR["Rerank head<br/>top 50"]
        RR ==> CAP["Max chunks per item"]
        CAP ==> MMRR["MMR diversity"]
    end

    subgraph RECOMMEND["HYBRID RECOMMENDATION"]
        direction LR
        CTX["Principal, kind<br/>and optional context"] ==> STATE["User state<br/>decayed history, negatives,<br/>profile vector, folded ALS vector"]
        STATE ==> CAND["Candidate sources<br/>context, content, collaborative,<br/>graph, popularity"]
        CAND ==> ELIG["Eligibility rules<br/>ACL, kind, status,<br/>assignment, dismissals"]
        ELIG ==> SIG["Signal scoring<br/>normalised per request"]
        SIG ==> BLENDING["Weighted blend<br/>weights renormalised<br/>for missing signals"]
        BLENDING ==> DIV["MMR on item vectors"]
        DIV ==> WHY["Reasons from top<br/>contributing signals"]
    end

    MMRR ==> CAND
    WHY ==> OUT["Ranked, explained results"]

    classDef default fill:none
    style RETRIEVAL fill:none
    style RECOMMEND fill:none
```

## How the hybrid engine works

### Hybrid retrieval

Project knowledge mixes natural language with exact identifiers: ticket keys, error codes, service names, acronyms and people's names. Dense embeddings handle paraphrase and concepts but miss rare tokens; BM25 handles exact tokens but misses paraphrase. Vantix runs both and fuses them.

1. **Dense recall** embeds the query (with an LRU query cache) and searches the vector index with the tenant, permission and business filters applied inside the index.
2. **Lexical recall** runs BM25 over the same filtered space. The in memory index is a true incremental inverted index; production uses OpenSearch with an English analyser.
3. **Weighted reciprocal rank fusion** combines the two rankings using ranks rather than raw scores, which avoids calibrating cosine similarity against BM25.
4. **Personal affinity** adds a configurable boost (default 0.15) for items close to the user in the work graph, so a search for "release checklist" favours the checklist of the user's own team.
5. **Reranking** rescores the head of the list. The default lexical reranker uses coverage, phrase, proximity and title features with no dependencies; a cross encoder or any hosted rerank endpoint can be configured.
6. **Per item capping and MMR** stop a single long document from occupying every slot and remove near duplicate passages.

Every hit returns its `fused`, `dense`, `sparse`, `coverage`, `relevance` and (when used) `rerank` and `affinity` signals for transparency and debugging.

### Hybrid recommendation

Recommendations are produced by multi source candidate generation followed by signal scoring, calibrated blending, business rules, diversification and explanation.

<table>
<tr><th>Signal</th><th>What it measures</th><th>How it is computed</th></tr>
<tr><td><code>content</code></td><td>Fit with the user's interests</td><td>Cosine similarity between the item vector and a real time profile vector built from the user's time decayed interaction history</td></tr>
<tr><td><code>context</code></td><td>Relevance to what the user is doing now</td><td>Hybrid retrieval over the current task, page or query, combined with vector similarity to the context</td></tr>
<tr><td><code>collaborative</code></td><td>What people with similar behaviour engage with</td><td>Implicit ALS (Hu, Koren and Volinsky) with a user vector folded in at request time from the latest interactions, so new activity matters immediately</td></tr>
<tr><td><code>graph</code></td><td>Structural closeness in the organisation's work</td><td>Local Personalized PageRank (push algorithm of Andersen, Chung and Lang) over users, items, projects and tags, seeded by the user and the context</td></tr>
<tr><td><code>popularity</code></td><td>What is trending in the workspace</td><td>Interaction weight with a seven day half life, normalised per tenant</td></tr>
<tr><td><code>urgency</code></td><td>Task pressure (tasks only)</td><td>Priority on a 0 to 4 scale combined with exponential due date proximity; blocked tasks are damped</td></tr>
<tr><td><code>evidence</code></td><td>Proven experience (experts only)</td><td>Relevance weighted count of closely related items the person authored or was assigned</td></tr>
<tr><td><code>activity</code></td><td>Availability proxy (experts only)</td><td>Time decayed recent activity of the person</td></tr>
</table>

Each signal is normalised per request, multiplied by the weight configured for the recommendation kind, and summed. When a signal is unavailable (for example `collaborative` for a brand new user), its weight is dropped and the remaining weights are renormalised, which handles cold start gracefully instead of shrinking every score.

Business rules are then applied: access control, kind filters, active status for tasks, exclusion of tasks assigned to other people (unless requested), exclusion of anything the user dismissed or downvoted, and, for discovery feeds without context, suppression of documents the user engaged with recently. Contextual panels always show the most relevant documents, even familiar ones. Finally, MMR on item vectors diversifies the list and the top contributing signals become plain language reasons such as *Similar to "Production deployment runbook", which you worked on recently*, *Directly linked to "Add automatic canary abort"* or *Due in 3 days*.

Four recommendation kinds are available:

<table>
<tr><th>Kind</th><th>Typical surface</th><th>Returns</th></tr>
<tr><td><code>documents</code></td><td>Home feed, knowledge sidebar, task panel</td><td>Knowledge items: documents, wiki pages, comments, files, messages, decisions</td></tr>
<tr><td><code>tasks</code></td><td>My work, up next, sprint planning</td><td>Active tasks assigned to the user or unassigned in the user's projects</td></tr>
<tr><td><code>similar</code></td><td>Related items section, duplicate detection</td><td>Items similar to a context item, by content, co engagement and graph proximity</td></tr>
<tr><td><code>experts</code></td><td>Assignee picker, "ask someone" prompts</td><td>People best placed to help with a context item or query</td></tr>
</table>

### Retrieval augmented generation

1. The question is combined with recent conversation turns and the context item title so that follow up questions retrieve correctly.
2. Hybrid retrieval runs with the caller's permissions and graph personalisation.
3. **Abstention gate.** If no passage clears either the dense similarity gate or the lexical coverage gate, Vantix returns an explicit "not found" answer instead of calling the model. This is the single most effective control against hallucination.
4. Passages are packed into a token budget with numbered source headers that carry title, section, kind, project and last updated date.
5. The system prompt requires a citation on every factual sentence, instructs the model to treat sources as data rather than instructions, and to prefer the most recent source when sources disagree.
6. Every citation in the output is validated against the sources actually supplied. Citations to non existent sources are removed, and the response reports exactly which sources were used and whether the answer is grounded.

Responses can be returned as a single JSON document or streamed as server sent events.

### Contextual assist

`POST /v1/assist` is the hybrid of both engines in one call and is designed for the task detail panel. Given the task a user is viewing and an optional question, Vantix computes in parallel a grounded answer, relevant documents, similar tasks (including completed ones, which often contain the solution) and suggested experts.

## Repository layout

```text
vantix/
  src/vantix/
    api/             FastAPI application, routes and public schemas
    domain/          Platform neutral domain model: Item, Interaction, Principal, Chunk
    embeddings/      Hashing, sentence transformers and OpenAI compatible embedders
    events/          Event contract and idempotent event processor
    evaluation/      Ranking metrics and offline evaluation harness
    index/           Vector and lexical index contracts, memory, Qdrant and OpenSearch adapters
    ingestion/       Chunker, change aware ingestion pipeline and background queue
    observability/   Structured logging and Prometheus metrics
    rag/             Prompts, language model providers and the RAG service
    recommender/     ALS, work graph, trainer, registry, scheduler and the engine
    retrieval/       Fusion, rerankers and the hybrid retriever
    security/        API key authentication, rate limiting, webhook signatures, ACL evaluation
    storage/         Relational schema and repository (SQLite or PostgreSQL)
    assist.py        Contextual assist orchestration
    cache.py         Versioned TTL cache
    cli.py           Command line interface
    client.py        Typed Python client for host platforms
    config.py        Settings, all read from VANTIX_ environment variables
    container.py     Composition root
  examples/
    data/                    Sample workspace: items, events and labelled queries
    generate_sample_data.py  Deterministic sample data generator
    host_integration.py      Reference integration for a host platform
  tests/             Unit, service, contract, API and real backend integration tests
  scripts/
    smoke_test.py            Post deployment smoke test used by CD and by the container job
    load_test.py             Concurrent load test with p95 latency and error budgets
    quality_gate.py          Model quality gate against agreed metric floors
    quality_thresholds.json  Floors for retrieval, recommendation and load metrics
    check_readme.py          README house style enforcement
    check_version.py         Release tag, package and chart version agreement
    pin_actions.py           Pins every GitHub Action to an immutable commit SHA
    mermaid/                 Renders every README diagram in light and dark themes
  deploy/helm/vantix/  Helm chart: API, training worker, migration job, HPA, PDB, NetworkPolicy
  requirements/      Hash verified runtime lock and development lock, generated by make lock
  .github/           CI, CD, release, deploy, security, CodeQL and nightly workflows
  Dockerfile         Hash verified, non root, read only friendly production image
  compose.yaml       Vantix with PostgreSQL, Qdrant and OpenSearch
  .env.example       Every configuration variable with defaults
  Makefile           Every CI check runnable locally (make help lists them)
```

## Quick start

The default configuration runs entirely offline with SQLite, in memory indexes, the hashing embedder, the lexical reranker and the extractive answerer. It needs Python 3.11 or newer and nothing else.

1. Create and activate a Python virtual environment, then install Vantix with the development tools.

   ```bash
   pip install ".[dev]"
   ```

2. Copy the example configuration.

   ```bash
   cp .env.example .env
   ```

3. Load the sample workspace (48 items across three projects and ninety days of activity from ten users) and train the first model.

   ```bash
   vantix demo
   ```

4. Start the API.

   ```bash
   vantix serve
   ```

5. Open the interactive API documentation at `http://localhost:8080/docs`, or call the API from Python.

   ```python
   import httpx

   client = httpx.Client(base_url="http://localhost:8080", headers={"Authorization": "Bearer demo_key"})
   principal = {"user_id": "amara", "project_ids": ["p_platform"], "group_ids": ["eng", "leads"]}

   answer = client.post("/v1/ask", json={"principal": principal, "question": "How long does a rollback take?"})
   print(answer.json()["answer"])

   tasks = client.post("/v1/recommendations", json={"principal": principal, "kind": "tasks", "k": 5})
   for rec in tasks.json()["recommendations"]:
       print(rec["title"], rec["score"], [reason["message"] for reason in rec["reasons"]])
   ```

6. Run the tests and the evaluation harness.

   ```bash
   pytest
   vantix evaluate retrieval examples/data/eval_queries.jsonl
   vantix evaluate recommender
   ```

## Production deployment

### Reference stack

`compose.yaml` starts the production shaped stack: Vantix, PostgreSQL 16, Qdrant and OpenSearch.

```bash
cp .env.example .env
docker compose up
```

The image is multi stage, runs as a non root user, exposes port 8080, declares a health check against `/v1/health` and stores model artifacts in the `/data` volume.

### Kubernetes with Helm

The chart in `deploy/helm/vantix` runs the single image in three roles:

<table>
<tr><th>Role</th><th>Kubernetes object</th><th>Behaviour</th></tr>
<tr><td>API</td><td>Deployment with HorizontalPodAutoscaler and PodDisruptionBudget</td><td><code>vantix serve</code>; zero downtime rolling updates, startup, liveness and readiness probes, topology spread across nodes</td></tr>
<tr><td>Training worker</td><td>Deployment with exactly one replica</td><td><code>vantix worker</code>; retrains changed tenants and persists models that API pods hot reload; liveness through <code>vantix health</code></td></tr>
<tr><td>Migrations</td><td>Job run as a pre install and pre upgrade hook</td><td><code>vantix migrate</code>; a failed migration fails the release before any new pod starts</td></tr>
</table>

Every pod runs as an unprivileged user with a read only root filesystem, all Linux capabilities dropped and the runtime default seccomp profile. A NetworkPolicy admits traffic only from namespaces and pods you list, and an optional ServiceMonitor scrapes `/v1/metrics`. Models live on a shared persistent volume (ReadWriteMany when pods can land on different nodes).

Secrets are never placed in values files. Create a Kubernetes Secret containing `VANTIX_API_KEYS`, `VANTIX_DATABASE_URL`, `VANTIX_WEBHOOK_SECRET` and any provider keys, ideally through your secrets manager or External Secrets, and set the chart value `existingSecret` to its name. Then run `helm upgrade` with the install option, your namespace and the values file for the environment (`values_staging.yaml` or `values_production.yaml`).

In practice the CD pipeline described below performs every deployment, pinning the image by digest and verifying its signature first. Released charts are also published to the GitHub container registry as OCI artifacts.

### Choosing model providers

The offline defaults are excellent for development and CI. For production quality, configure a neural embedding model and a stronger reranker and language model. All providers are selected by environment variables.

**Hosted embeddings through any OpenAI compatible endpoint** (OpenAI, Azure gateways, vLLM, Text Embeddings Inference, Ollama):

```bash
VANTIX_EMBEDDING_PROVIDER=openai_compatible
VANTIX_EMBEDDING_BASE_URL=https://your.embedding.gateway/v1
VANTIX_EMBEDDING_API_KEY=your_key
VANTIX_EMBEDDING_MODEL=your_embedding_model
VANTIX_EMBEDDING_DIM=1536
```

**Local embeddings and cross encoder reranking in process** (install with `pip install ".[models]"`):

```bash
VANTIX_EMBEDDING_PROVIDER=sentence_transformers
VANTIX_EMBEDDING_MODEL=your_sentence_transformers_model
VANTIX_EMBEDDING_DIM=384
VANTIX_RERANKER_PROVIDER=cross_encoder
VANTIX_RERANKER_MODEL=your_cross_encoder_model
```

**Hosted reranking** through any endpoint that accepts the common `query`, `documents`, `top_n` request shape:

```bash
VANTIX_RERANKER_PROVIDER=http
VANTIX_RERANKER_URL=https://your.reranker/v1/rerank
VANTIX_RERANKER_API_KEY=your_key
```

**Generation** with Anthropic or any OpenAI compatible chat endpoint:

```bash
VANTIX_LLM_PROVIDER=anthropic
VANTIX_LLM_MODEL=your_claude_model
VANTIX_LLM_API_KEY=your_key
```

```bash
VANTIX_LLM_PROVIDER=openai_compatible
VANTIX_LLM_BASE_URL=https://your.llm.gateway/v1
VANTIX_LLM_MODEL=your_chat_model
VANTIX_LLM_API_KEY=your_key
```

`VANTIX_EMBEDDING_DIM` must equal the dimension the model produces. Vantix validates this on every embedding call and Qdrant validates it against the collection on startup.

### Scaling to a very large knowledge base

* **Vectors.** Qdrant collections are created with cosine distance, HNSW (m 16, ef construct 200), int8 scalar quantisation kept in RAM with rescoring, original vectors on disk and payload on disk. This keeps tens of millions of chunks affordable. `tenant_id` is a tenant aware payload index so each tenant's vectors are colocated, and every ACL field is indexed so filtered search stays fast. Shard and replicate the collection through Qdrant's own cluster configuration.
* **Lexical.** The OpenSearch index uses three primary shards and one replica by default, a strict mapping and keyword fields for every filter. Increase shards in proportion to corpus size.
* **System of record.** Use PostgreSQL in production. Tables carry composite primary keys on tenant and identifier plus indexes for every access pattern used by the service. Embeddings are stored as compact float32 blobs, so indexes can always be rebuilt without calling the embedding provider again.
* **Ingestion cost.** Unchanged items are skipped by fingerprint, metadata only changes (status, assignee, ACL) refresh index payloads without re embedding, and embedding calls are batched across items.
* **API replicas.** With Qdrant, OpenSearch and PostgreSQL the API is stateless with respect to request handling. Run several replicas behind a load balancer, set `VANTIX_WORKERS` per container, and size the ingestion thread pool with `VANTIX_INGESTION_WORKERS`.
* **Recommender training.** Run exactly one dedicated training worker (`vantix worker`). It polls PostgreSQL for tenants whose content version or interaction count changed since their last model, retrains only those, and persists models to the shared `VANTIX_MODEL_DIR`. API replicas run with `VANTIX_REC_TRAINING_INTERVAL_SECONDS=0` and hot reload newer models from that directory within `VANTIX_MODEL_REFRESH_SECONDS` (30 by default), with no restarts. The Helm chart wires this up for you.
* **Schema migrations.** `vantix migrate` creates or upgrades the schema. On PostgreSQL it holds an advisory lock, so a migration job and many starting replicas can never race. The Helm chart runs it as a hook before every install and upgrade.
* **Shared rate limiting.** The built in limiter is per process. When running many replicas, enforce tenant quotas at the API gateway as well.
* **Memory backends.** The in memory vector and lexical indexes are intended for development, tests and single node deployments up to a few hundred thousand chunks. They live inside one process, so `vantix serve` runs a single worker process whenever a memory backend is configured. They are rebuilt from the system of record automatically on startup.

## Integration guide

This section is the contract between a host platform such as ProWin and Vantix. Following it step by step gives a complete, secure and production ready integration.

### Integration model

The host platform keeps full ownership of its data, users and permissions. It integrates through three channels:

<table>
<tr><th>Channel</th><th>Direction</th><th>Purpose</th><th>Endpoints</th></tr>
<tr><td>Content and activity</td><td>Host to Vantix, asynchronous</td><td>Keep Vantix current with every item change and every meaningful user interaction</td><td><code>POST /v1/events</code>, <code>POST /v1/items</code>, <code>DELETE /v1/items/{item_id}</code></td></tr>
<tr><td>Intelligence</td><td>Host to Vantix, synchronous</td><td>Search, answers, recommendations and contextual assist for a specific user</td><td><code>POST /v1/search</code>, <code>POST /v1/ask</code>, <code>POST /v1/recommendations</code>, <code>POST /v1/assist</code></td></tr>
<tr><td>Feedback</td><td>Host to Vantix, synchronous</td><td>Report clicks, votes, dismissals and bookmarks on what Vantix showed</td><td><code>POST /v1/feedback</code></td></tr>
</table>

Browsers and mobile apps should never call Vantix directly. The host backend calls Vantix server to server, attaches the API key, and builds the principal from the authenticated session. This keeps API keys secret and makes the principal trustworthy.

### Step 1: Provision tenants and API keys

Vantix is multitenant. A tenant is an isolated workspace: data, indexes, models and caches never cross tenant boundaries. Map one Vantix tenant to one customer workspace in the host platform.

Keys are configured as comma separated `tenant_id:api_key` pairs. Generate long random keys in your secrets manager and inject them as environment variables.

```bash
VANTIX_API_KEYS=workspace_acme:k_9f1c7e...,workspace_globex:k_51ab0d...
VANTIX_ADMIN_API_KEYS=ops:k_admin_77e2...
```

Every request authenticates with the standard `Authorization` header in the form `Bearer <api_key>`. The tenant is derived from the key, so it can never be spoofed in a request body. Administrative keys can additionally operate on any tenant through the `tenant_id` query parameter of the `/v1/admin` endpoints.

### Step 2: Map your domain onto Vantix items

Everything a user can search, read or work on becomes an `Item`. The mapping below shows a typical ProWin mapping; any project management platform maps the same way.

<table>
<tr><th>Item field</th><th>Type</th><th>Meaning</th><th>Typical ProWin source</th></tr>
<tr><td><code>id</code></td><td>string, required</td><td>Stable unique identifier within the tenant</td><td>Task key, page id, comment id, file id</td></tr>
<tr><td><code>kind</code></td><td>enum</td><td><code>task</code>, <code>document</code>, <code>wiki_page</code>, <code>comment</code>, <code>file</code>, <code>message</code>, <code>decision</code></td><td>Entity type</td></tr>
<tr><td><code>title</code></td><td>string</td><td>Short title, also used as a contextual chunk header</td><td>Task summary, page title</td></tr>
<tr><td><code>body</code></td><td>string</td><td>Full text. Markdown headings are respected by the chunker; HTML is stripped</td><td>Description, page content, extracted file text</td></tr>
<tr><td><code>url</code></td><td>string</td><td>Deep link back into the host user interface</td><td>Canonical item URL</td></tr>
<tr><td><code>project_id</code></td><td>string</td><td>Owning project, used for access control, filters and the work graph</td><td>Project or space id</td></tr>
<tr><td><code>author_id</code></td><td>string</td><td>Creator. Authors can always see their own items</td><td>Reporter or creator</td></tr>
<tr><td><code>parent_id</code></td><td>string</td><td>Parent item, for example the task a comment belongs to</td><td>Parent task, epic or page</td></tr>
<tr><td><code>assignee_ids</code></td><td>list of strings</td><td>People assigned. Drives task eligibility, the work graph and expertise</td><td>Assignees</td></tr>
<tr><td><code>linked_item_ids</code></td><td>list of strings</td><td>Explicit relations such as "relates to", "blocks" and attached pages</td><td>Issue links, page mentions</td></tr>
<tr><td><code>tags</code></td><td>list of strings</td><td>Labels or components, normalised to lower case</td><td>Labels, components</td></tr>
<tr><td><code>status</code></td><td>enum</td><td><code>open</code>, <code>in_progress</code>, <code>blocked</code>, <code>in_review</code>, <code>done</code>, <code>cancelled</code>. The last two are inactive</td><td>Workflow status mapped to these six values</td></tr>
<tr><td><code>priority</code></td><td>integer 0 to 4</td><td>0 none, 1 low, 2 medium, 3 high, 4 urgent</td><td>Priority field</td></tr>
<tr><td><code>due_at</code></td><td>ISO 8601 datetime</td><td>Due date, used for urgency</td><td>Due date</td></tr>
<tr><td><code>created_at</code>, <code>updated_at</code></td><td>ISO 8601 datetime</td><td>Timestamps. Recency and source freshness in answers use <code>updated_at</code></td><td>Audit timestamps</td></tr>
<tr><td><code>acl</code></td><td>object</td><td>Visibility rules, described below</td><td>Permission scheme</td></tr>
<tr><td><code>metadata</code></td><td>object</td><td>Free form attributes stored with the item</td><td>Custom fields</td></tr>
</table>

Item payloads are validated strictly: unknown fields are rejected with status 422 so mapping mistakes surface immediately.

**Access control.** Each item carries an `acl` object. Visibility is evaluated against the principal in this order:

<table>
<tr><th>Rule</th><th>Grants access when</th></tr>
<tr><td>Administrator</td><td>The principal has <code>is_admin</code> set to true</td></tr>
<tr><td>Author</td><td>The principal is the item's <code>author_id</code></td></tr>
<tr><td>Explicit user grant</td><td>The principal's <code>user_id</code> is in <code>acl.allowed_user_ids</code></td></tr>
<tr><td>Explicit group grant</td><td>Any of the principal's <code>group_ids</code> is in <code>acl.allowed_group_ids</code></td></tr>
<tr><td><code>visibility</code> is <code>tenant</code></td><td>Always, for every user in the tenant</td></tr>
<tr><td><code>visibility</code> is <code>project</code> (default)</td><td>The item's <code>project_id</code> is in the principal's <code>project_ids</code>, or the item has no project</td></tr>
<tr><td><code>visibility</code> is <code>restricted</code></td><td>Only through one of the rules above</td></tr>
</table>

When permissions change in the host (an item moves project, a page becomes private, a group is granted access), send an `item.upserted` event with the new `acl`. Vantix detects that only metadata changed and refreshes the index payloads within seconds, without re embedding.

### Step 3: Build the principal on every read

Every intelligence call includes a `principal` describing the end user, built by the host backend from its own session and membership data:

```json
{
  "user_id": "amara",
  "project_ids": ["p_platform", "p_data"],
  "group_ids": ["eng", "leads"],
  "is_admin": false
}
```

Include every project and group the user can currently access. Cache memberships in the host for a short time if they are expensive to compute, but never accept a principal from the client side.

### Step 4: Backfill existing content

Send existing items in batches of up to 1000. By default the call returns status 202 with a job identifier and indexing happens in the background.

```python
from vantix.client import VantixClient

client = VantixClient("https://vantix.internal", api_key="k_9f1c7e")

job = client.upsert_items(items_batch)
status = client.job(job["job_id"])
```

Poll `GET /v1/jobs/{job_id}` until `status` is `completed` or `completed_with_errors`. Jobs report `total`, `processed`, `failed` and the first errors with the failing item ids. Other statuses are `queued`, `running`, `failed` and `rejected` (when the queue is full the API returns status 503 and the caller should retry with backoff).

Set `"wait": true` in the request body to index synchronously and receive per item results with status 200. This is convenient for small batches and tests.

For very large initial loads, the CLI ingests JSON Lines or JSON array files directly into the system of record without going through HTTP:

```bash
vantix ingest exports/items.jsonl workspace_acme
vantix events exports/events.jsonl workspace_acme
vantix train workspace_acme
```

### Step 5: Stream changes with signed, idempotent events

After the backfill, keep Vantix current by forwarding changes as events to `POST /v1/events`. Each request carries a batch of up to 1000 events.

<table>
<tr><th>Event type</th><th>When to send</th><th><code>data</code> payload</th></tr>
<tr><td><code>item.upserted</code></td><td>An item is created or any of its fields change, including status, assignees and permissions</td><td>The full <code>Item</code> object</td></tr>
<tr><td><code>item.deleted</code></td><td>An item is deleted or archived out of scope</td><td><code>{"id": "..."}</code></td></tr>
<tr><td><code>interaction.recorded</code></td><td>A user meaningfully interacts with an item</td><td><code>user_id</code>, <code>item_id</code>, <code>type</code>, optional <code>project_id</code>, <code>query</code>, <code>value</code></td></tr>
</table>

```json
{
  "events": [
    {
      "event_id": "evt_000184213",
      "type": "item.upserted",
      "data": {
        "id": "t_canary_alerts",
        "kind": "task",
        "title": "Add automatic canary abort on error spike",
        "body": "Abort the canary when the error rate exceeds one percent for five minutes.",
        "project_id": "p_platform",
        "status": "in_review",
        "priority": 3,
        "assignee_ids": ["amara"],
        "linked_item_ids": ["kb_deploy_runbook", "kb_monitoring"],
        "acl": {"visibility": "project"}
      }
    },
    {
      "event_id": "evt_000184214",
      "type": "interaction.recorded",
      "data": {"user_id": "ben", "item_id": "kb_deploy_runbook", "type": "view", "project_id": "p_platform"}
    }
  ]
}
```

Each event may also carry `occurred_at` as an ISO 8601 timestamp; when it is omitted, the time of receipt is used. Events in a batch are applied in `occurred_at` order, and for the same item the last change in the batch wins.

**Delivery guarantees.** Delivery is at least once. Vantix deduplicates on `event_id`, so the host can retry any batch safely. Use a stable identifier from the host's outbox row as the `event_id`.

**Recommended pattern: transactional outbox.** Write each event to an outbox table in the same database transaction as the business change, and run a relay that reads unsent rows in order, posts them in batches, and marks them sent on a 202 response. This guarantees that no change is ever lost and that Vantix never sees a change that was rolled back.

**Signing.** When `VANTIX_WEBHOOK_SECRET` is set, every events request must carry a `Signature` header with the value `t=<unix_seconds>,v1=<hex_digest>`, where the digest is HMAC SHA256 over the string `<unix_seconds>.` followed by the exact raw request body, keyed with the shared secret. Requests older than `VANTIX_WEBHOOK_TOLERANCE_SECONDS` (300 by default) are rejected to prevent replay. The Python client signs automatically:

```python
client = VantixClient("https://vantix.internal", api_key="k_9f1c7e", webhook_secret="whsec_live_value")
result = client.send_events(events)
```

For hosts written in other languages the algorithm is:

```python
import hashlib
import hmac
import json
import time

import httpx

body = json.dumps({"events": events}).encode()
timestamp = int(time.time())
digest = hmac.new(secret.encode(), str(timestamp).encode() + b"." + body, hashlib.sha256).hexdigest()
headers = {"Authorization": "Bearer " + api_key, "Signature": "t=" + str(timestamp) + ",v1=" + digest}
httpx.post("https://vantix.internal/v1/events", content=body, headers=headers)
```

The response reports `accepted`, `duplicates`, `interactions_recorded`, `items_queued`, `deletes_queued` and, when content was queued, the `job_id`.

### Step 6: Record interactions

Interactions are the fuel for personalisation. Send them as `interaction.recorded` events from the host backend whenever a user does something meaningful. Each type carries a default implicit feedback weight; supply `value` to override it.

<table>
<tr><th>Interaction type</th><th>Default weight</th><th>Send when the user</th></tr>
<tr><td><code>view</code></td><td>1</td><td>Opens an item and stays on it (for example longer than five seconds)</td></tr>
<tr><td><code>open</code></td><td>1</td><td>Opens an attachment or expands a preview</td></tr>
<tr><td><code>search_click</code></td><td>2</td><td>Clicks a search result (include the <code>query</code>)</td></tr>
<tr><td><code>recommendation_click</code></td><td>2</td><td>Clicks a recommendation</td></tr>
<tr><td><code>create</code></td><td>5</td><td>Creates the item</td></tr>
<tr><td><code>edit</code></td><td>4</td><td>Edits the item</td></tr>
<tr><td><code>comment</code></td><td>3</td><td>Comments on the item</td></tr>
<tr><td><code>assign</code></td><td>4</td><td>Takes or is given the task</td></tr>
<tr><td><code>complete</code></td><td>5</td><td>Completes the task</td></tr>
<tr><td><code>bookmark</code></td><td>4</td><td>Bookmarks, stars or follows the item</td></tr>
<tr><td><code>share</code></td><td>3</td><td>Shares the item</td></tr>
<tr><td><code>upvote</code></td><td>3</td><td>Gives positive feedback</td></tr>
<tr><td><code>downvote</code></td><td>negative 3</td><td>Gives negative feedback; the item is suppressed for that user</td></tr>
<tr><td><code>dismiss</code></td><td>negative 2</td><td>Dismisses a suggestion; the item is suppressed for that user</td></tr>
</table>

Interaction weights decay with a configurable half life (30 days by default) so that recent behaviour dominates without discarding long term interests.

### Step 7: Wire the intelligence into the user interface

<table>
<tr><th>Host surface</th><th>Vantix call</th><th>Notes</th></tr>
<tr><td>Global search bar</td><td><code>POST /v1/search</code></td><td>Personalised by default; filter by <code>item_kinds</code>, <code>project_ids</code>, <code>active_tasks_only</code>, <code>updated_after</code></td></tr>
<tr><td>Ask assistant</td><td><code>POST /v1/ask</code> with <code>"stream": true</code></td><td>Render tokens as they arrive and show the cited sources as links</td></tr>
<tr><td>Home "for you" feed</td><td><code>POST /v1/recommendations</code>, kind <code>documents</code></td><td>Discovery feed that avoids repeating recently read items</td></tr>
<tr><td>My work, up next</td><td><code>POST /v1/recommendations</code>, kind <code>tasks</code></td><td>Active tasks assigned to the user or unassigned in their projects</td></tr>
<tr><td>Task detail side panel</td><td><code>POST /v1/assist</code> with <code>item_id</code></td><td>Documents, similar tasks and experts in one call; add <code>question</code> for an inline answer</td></tr>
<tr><td>Related items section</td><td><code>POST /v1/recommendations</code>, kind <code>similar</code></td><td>Pass <code>context_item_id</code>; restrict with <code>item_kinds</code></td></tr>
<tr><td>Assignee picker</td><td><code>POST /v1/recommendations</code>, kind <code>experts</code></td><td>Pass <code>context_item_id</code> and the eligible team as <code>candidate_user_ids</code></td></tr>
<tr><td>New task form, duplicate hint</td><td><code>POST /v1/recommendations</code>, kind <code>similar</code></td><td>Pass the draft title as <code>context_query</code> and <code>item_kinds</code> of <code>task</code></td></tr>
</table>

Always render the `reasons` returned with each recommendation. Explanations measurably increase trust and click through, and they make the system auditable.

### Step 8: Close the loop with feedback

Report what users do with Vantix output through `POST /v1/feedback`:

```json
{
  "principal": {"user_id": "amara", "project_ids": ["p_platform"]},
  "target_id": "kb_monitoring",
  "signal": "dismiss",
  "surface": "recommendation"
}
```

`signal` is one of `click`, `upvote`, `downvote`, `dismiss` or `bookmark`, and `surface` is one of `recommendation`, `search`, `answer` or `assist`. Dismissals and downvotes take effect on the very next request for that user; positive feedback feeds the content profile immediately and the collaborative model at the next training run. Pass your own `event_id` to make feedback idempotent across retries.

### Step 9: Keep models fresh

In a single process deployment, interactions and content changes mark a tenant as dirty and the background scheduler retrains dirty tenants every `VANTIX_REC_TRAINING_INTERVAL_SECONDS` (one hour by default). In a multi replica deployment the dedicated training worker detects changes through the database instead, and API replicas hot reload the models it persists. Either way the new model is swapped in atomically and persisted to `VANTIX_MODEL_DIR` for warm restarts. Between trainings, the content profile and the folded in collaborative vector already reflect each user's newest interactions. Trigger an immediate training run with `POST /v1/admin/train` or `vantix train TENANT`.

### Serving several Phexara products

Vantix is product neutral. Two patterns work well:

* **Shared tenant per customer.** When several products serve the same customer workspace (for example ProWin tasks and a separate documentation product), send all their items into the same tenant with globally unique item identifiers (for example prefixed by product). Users then get cross product search, answers and recommendations, and the work graph links people across products.
* **Separate tenants per product.** When products serve different audiences or have different compliance boundaries, give each product its own tenant and API keys. Data, models and caches remain fully isolated.

## API reference

The full OpenAPI specification is served by every instance at `/openapi.json`, with interactive documentation at `/docs`. All endpoints except health, readiness and metrics require the `Authorization` bearer header.

<table>
<tr><th>Method</th><th>Path</th><th>Purpose</th><th>Success</th></tr>
<tr><td>GET</td><td><code>/v1/health</code></td><td>Liveness probe and version</td><td>200</td></tr>
<tr><td>GET</td><td><code>/v1/ready</code></td><td>Readiness of database, vector and lexical indexes</td><td>200 or 503</td></tr>
<tr><td>GET</td><td><code>/v1/metrics</code></td><td>Prometheus metrics</td><td>200</td></tr>
<tr><td>POST</td><td><code>/v1/items</code></td><td>Upsert up to 1000 items, queued or synchronous</td><td>202 or 200</td></tr>
<tr><td>GET</td><td><code>/v1/items/{item_id}</code></td><td>Read an item as stored</td><td>200</td></tr>
<tr><td>DELETE</td><td><code>/v1/items/{item_id}</code></td><td>Delete an item and remove all of its chunks from every index</td><td>200</td></tr>
<tr><td>GET</td><td><code>/v1/jobs/{job_id}</code></td><td>Ingestion job progress</td><td>200</td></tr>
<tr><td>POST</td><td><code>/v1/events</code></td><td>Signed, idempotent event batch</td><td>202</td></tr>
<tr><td>POST</td><td><code>/v1/search</code></td><td>Personalised hybrid search</td><td>200</td></tr>
<tr><td>POST</td><td><code>/v1/ask</code></td><td>Grounded answer with citations, JSON or streamed</td><td>200</td></tr>
<tr><td>POST</td><td><code>/v1/recommendations</code></td><td>Documents, tasks, similar items or experts</td><td>200</td></tr>
<tr><td>POST</td><td><code>/v1/assist</code></td><td>Contextual assist for a task panel</td><td>200</td></tr>
<tr><td>POST</td><td><code>/v1/feedback</code></td><td>Record user feedback on Vantix output</td><td>200</td></tr>
<tr><td>DELETE</td><td><code>/v1/users/{user_id}/interactions</code></td><td>Right to erasure: delete a user's entire interaction history</td><td>200</td></tr>
<tr><td>POST</td><td><code>/v1/admin/train</code></td><td>Retrain the tenant's recommender now</td><td>200</td></tr>
<tr><td>POST</td><td><code>/v1/admin/reindex</code></td><td>Rebuild indexes from the system of record; <code>full=true</code> rechunks and re embeds</td><td>200</td></tr>
<tr><td>GET</td><td><code>/v1/admin/stats</code></td><td>Item, chunk, interaction and index counts, model and provider details</td><td>200</td></tr>
</table>

### Search

```json
{
  "principal": {"user_id": "amara", "project_ids": ["p_platform"], "group_ids": ["eng"]},
  "query": "refresh token reused",
  "k": 5,
  "item_kinds": ["document", "wiki_page"],
  "personalize": true
}
```

Each hit returns `item_id`, `chunk_id`, `kind`, `title`, `url`, `project_id`, `heading`, a `snippet` of the matching passage, the final `score` and the individual `signals`.

### Ask

```json
{
  "principal": {"user_id": "amara", "project_ids": ["p_platform"]},
  "question": "How long does a rollback take?",
  "history": [{"role": "user", "content": "We had a failed release this morning"}],
  "context_item_id": "t_canary_alerts",
  "k": 8,
  "stream": false
}
```

Response:

```json
{
  "answer": "Run the rollback job in the release pipeline and select the previous image tag. Rollback takes about four minutes. [1]",
  "citations": [
    {
      "index": 1,
      "item_id": "kb_deploy_runbook",
      "chunk_id": "kb_deploy_runbook::3",
      "title": "Production deployment runbook",
      "url": "https://app.example.com/wiki_pages/kb_deploy_runbook",
      "snippet": "Run the rollback job in the release pipeline...",
      "score": 0.71
    }
  ],
  "grounded": true,
  "abstained": false,
  "model": "your_claude_model",
  "usage": {"input_tokens": 1830, "output_tokens": 64}
}
```

When `abstained` is true, no source was relevant enough and the answer explains that the accessible knowledge base does not cover the question. When `grounded` is false and `abstained` is false, the model answered without valid citations; hosts may choose to display such answers with a warning or not at all.

**Streaming.** With `"stream": true`, the response is a server sent events stream with three event names: `sources` (the numbered sources, sent first), `token` (incremental text) and `done` (the final validated answer, the citations actually used, and the `grounded`, `abstained` and `model` fields). An `error` event is sent if generation fails mid stream.

### Recommendations

```json
{
  "principal": {"user_id": "hugo", "project_ids": ["p_mobile"]},
  "kind": "experts",
  "k": 3,
  "context_item_id": "t_refresh_token_bug",
  "candidate_user_ids": ["chloe", "dev", "grace", "ivy"]
}
```

<table>
<tr><th>Field</th><th>Applies to</th><th>Description</th></tr>
<tr><td><code>kind</code></td><td>all</td><td><code>documents</code>, <code>tasks</code>, <code>similar</code> or <code>experts</code></td></tr>
<tr><td><code>k</code></td><td>all</td><td>Number of results, 1 to 100</td></tr>
<tr><td><code>context_item_id</code></td><td>all</td><td>The item the user is looking at; required for <code>similar</code></td></tr>
<tr><td><code>context_query</code></td><td>all</td><td>Free text context such as a draft title or a search query</td></tr>
<tr><td><code>context_project_id</code></td><td>all</td><td>Project the user is browsing; seeds the work graph</td></tr>
<tr><td><code>item_kinds</code></td><td>items</td><td>Restrict result kinds</td></tr>
<tr><td><code>project_ids</code></td><td>items</td><td>Restrict results to these projects</td></tr>
<tr><td><code>exclude_ids</code></td><td>items</td><td>Items already on screen</td></tr>
<tr><td><code>include_assigned_to_others</code></td><td>tasks</td><td>Also return tasks assigned to other people</td></tr>
<tr><td><code>candidate_user_ids</code></td><td>experts</td><td>Restrict suggestions to these people, for example the project team</td></tr>
<tr><td><code>include_self</code></td><td>experts</td><td>Allow the requesting user in the results</td></tr>
<tr><td><code>diversify</code></td><td>all</td><td>Apply MMR diversification (default true)</td></tr>
</table>

Response:

```json
{
  "kind": "experts",
  "recommendations": [
    {
      "target_id": "dev",
      "target_type": "user",
      "score": 0.83,
      "reasons": [
        {"code": "evidence", "message": "Worked on 2 closely related items, including \"Authentication and OAuth token handling\"", "weight": 0.25},
        {"code": "graph", "message": "Closely connected to this work and project", "weight": 0.3}
      ],
      "signals": {"content": 0.71, "graph": 1.0, "evidence": 1.0, "activity": 0.64}
    }
  ],
  "model_trained_at": "time of the last training run"
}
```

Item recommendations have the same shape with `target_type` set to the item kind and `title` and `url` populated.

### Assist

```json
{
  "principal": {"user_id": "amara", "project_ids": ["p_platform"], "group_ids": ["eng"]},
  "item_id": "t_canary_alerts",
  "question": "When should I stop a canary rollout?",
  "k": 5
}
```

The response contains `answer` (the same shape as the ask response, or null when no question is given), `documents`, `similar_tasks` and `experts`.

### Errors

Errors use one consistent body and stable machine readable codes. The request identifier is also returned in the response headers and appears in every log line for that request.

```json
{"code": "rate_limited", "message": "Rate limit exceeded", "details": {"retry_after_seconds": 0.4}, "request_id": "c0a8f3d2e1"}
```

<table>
<tr><th>Status</th><th>Code</th><th>Meaning</th></tr>
<tr><td>401</td><td><code>unauthenticated</code></td><td>Missing or invalid API key, or an invalid, missing or expired webhook signature</td></tr>
<tr><td>403</td><td><code>forbidden</code></td><td>An administrative key is required</td></tr>
<tr><td>404</td><td><code>not_found</code></td><td>Item or job not found in the caller's tenant</td></tr>
<tr><td>422</td><td><code>validation_failed</code> or a schema error list</td><td>The request body, or an event in a batch, does not match the schema</td></tr>
<tr><td>429</td><td><code>rate_limited</code></td><td>Too many requests for the key; a standard retry header gives the wait in seconds</td></tr>
<tr><td>502</td><td><code>upstream_error</code></td><td>A model provider or index failed after bounded retries with exponential backoff</td></tr>
<tr><td>503</td><td><code>queue_full</code></td><td>The ingestion queue is at capacity; retry with backoff</td></tr>
</table>

## Configuration reference

All settings are environment variables prefixed with `VANTIX_`, and can also be placed in a `.env` file. `.env.example` lists every variable with its default.

<table>
<tr><th>Variable</th><th>Default</th><th>Description</th></tr>
<tr><td colspan="3"><b>Service</b></td></tr>
<tr><td><code>VANTIX_ENVIRONMENT</code></td><td><code>development</code></td><td>Free text environment name included in logs</td></tr>
<tr><td><code>VANTIX_HOST</code>, <code>VANTIX_PORT</code></td><td><code>0.0.0.0</code>, <code>8080</code></td><td>Bind address and port</td></tr>
<tr><td><code>VANTIX_WORKERS</code></td><td><code>1</code></td><td>Server worker processes per container</td></tr>
<tr><td><code>VANTIX_LOG_LEVEL</code>, <code>VANTIX_LOG_JSON</code></td><td><code>INFO</code>, <code>true</code></td><td>Log level and structured JSON output</td></tr>
<tr><td><code>VANTIX_CORS_ORIGINS</code></td><td>empty</td><td>Comma separated origins; leave empty for server to server use</td></tr>
<tr><td colspan="3"><b>Security</b></td></tr>
<tr><td><code>VANTIX_API_KEYS</code></td><td><code>demo:demo_key</code></td><td>Comma separated <code>tenant_id:api_key</code> pairs. Replace in every real deployment</td></tr>
<tr><td><code>VANTIX_ADMIN_API_KEYS</code></td><td>empty</td><td>Administrative keys in the same format</td></tr>
<tr><td><code>VANTIX_WEBHOOK_SECRET</code></td><td>empty</td><td>Shared secret for event signatures; signatures are required when set</td></tr>
<tr><td><code>VANTIX_WEBHOOK_TOLERANCE_SECONDS</code></td><td><code>300</code></td><td>Maximum signature age</td></tr>
<tr><td><code>VANTIX_RATE_LIMIT_PER_MINUTE</code></td><td><code>1200</code></td><td>Requests per minute per key; 0 disables</td></tr>
<tr><td colspan="3"><b>Storage</b></td></tr>
<tr><td><code>VANTIX_DATABASE_URL</code></td><td><code>sqlite:///./data/vantix.db</code></td><td>SQLAlchemy URL. Production: <code>postgresql+psycopg://user:password@host:5432/vantix</code></td></tr>
<tr><td><code>VANTIX_MODEL_DIR</code></td><td><code>./data/models</code></td><td>Directory for persisted recommender models</td></tr>
<tr><td colspan="3"><b>Embeddings</b></td></tr>
<tr><td><code>VANTIX_EMBEDDING_PROVIDER</code></td><td><code>hashing</code></td><td><code>hashing</code>, <code>sentence_transformers</code> or <code>openai_compatible</code></td></tr>
<tr><td><code>VANTIX_EMBEDDING_MODEL</code></td><td>a small local model</td><td>Model identifier for the provider</td></tr>
<tr><td><code>VANTIX_EMBEDDING_DIM</code></td><td><code>384</code></td><td>Vector dimension; must match the model</td></tr>
<tr><td><code>VANTIX_EMBEDDING_BASE_URL</code>, <code>VANTIX_EMBEDDING_API_KEY</code></td><td>OpenAI URL, empty</td><td>Endpoint and key for <code>openai_compatible</code></td></tr>
<tr><td><code>VANTIX_EMBEDDING_BATCH_SIZE</code>, <code>VANTIX_EMBEDDING_TIMEOUT_SECONDS</code></td><td><code>64</code>, <code>30</code></td><td>Batching and timeout</td></tr>
<tr><td colspan="3"><b>Indexes</b></td></tr>
<tr><td><code>VANTIX_VECTOR_BACKEND</code></td><td><code>memory</code></td><td><code>memory</code> or <code>qdrant</code></td></tr>
<tr><td><code>VANTIX_QDRANT_URL</code>, <code>VANTIX_QDRANT_API_KEY</code></td><td><code>http://localhost:6333</code>, empty</td><td>Qdrant connection</td></tr>
<tr><td><code>VANTIX_QDRANT_COLLECTION</code>, <code>VANTIX_QDRANT_HNSW_EF</code></td><td><code>vantix_chunks</code>, <code>128</code></td><td>Collection name and search breadth</td></tr>
<tr><td><code>VANTIX_LEXICAL_BACKEND</code></td><td><code>memory</code></td><td><code>memory</code> or <code>opensearch</code></td></tr>
<tr><td><code>VANTIX_OPENSEARCH_URL</code>, <code>VANTIX_OPENSEARCH_INDEX</code></td><td><code>http://localhost:9200</code>, <code>vantix_chunks</code></td><td>OpenSearch connection and index</td></tr>
<tr><td><code>VANTIX_OPENSEARCH_USERNAME</code>, <code>VANTIX_OPENSEARCH_PASSWORD</code></td><td>empty</td><td>Basic authentication</td></tr>
<tr><td colspan="3"><b>Chunking</b></td></tr>
<tr><td><code>VANTIX_CHUNK_TARGET_TOKENS</code></td><td><code>350</code></td><td>Target chunk size</td></tr>
<tr><td><code>VANTIX_CHUNK_OVERLAP_TOKENS</code></td><td><code>50</code></td><td>Overlap between consecutive chunks</td></tr>
<tr><td><code>VANTIX_CHUNK_MIN_TOKENS</code></td><td><code>40</code></td><td>Small trailing chunks are merged below this size</td></tr>
<tr><td colspan="3"><b>Retrieval</b></td></tr>
<tr><td><code>VANTIX_RETRIEVAL_DENSE_K</code>, <code>VANTIX_RETRIEVAL_SPARSE_K</code></td><td><code>100</code>, <code>100</code></td><td>Recall depth per retriever</td></tr>
<tr><td><code>VANTIX_RETRIEVAL_FUSION_K</code></td><td><code>60</code></td><td>RRF constant</td></tr>
<tr><td><code>VANTIX_RETRIEVAL_DENSE_WEIGHT</code>, <code>VANTIX_RETRIEVAL_SPARSE_WEIGHT</code></td><td><code>1.0</code>, <code>1.0</code></td><td>Fusion weights</td></tr>
<tr><td><code>VANTIX_RETRIEVAL_PERSONAL_WEIGHT</code></td><td><code>0.15</code></td><td>Graph affinity boost; 0 disables personalisation</td></tr>
<tr><td><code>VANTIX_RETRIEVAL_MMR_LAMBDA</code></td><td><code>0.75</code></td><td>Relevance versus diversity balance</td></tr>
<tr><td><code>VANTIX_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT</code></td><td><code>3</code></td><td>Per item cap</td></tr>
<tr><td><code>VANTIX_RETRIEVAL_CACHE_TTL_SECONDS</code></td><td><code>120</code></td><td>Result cache lifetime; 0 disables</td></tr>
<tr><td colspan="3"><b>Reranking</b></td></tr>
<tr><td><code>VANTIX_RERANKER_PROVIDER</code></td><td><code>lexical</code></td><td><code>none</code>, <code>lexical</code>, <code>cross_encoder</code> or <code>http</code></td></tr>
<tr><td><code>VANTIX_RERANKER_MODEL</code>, <code>VANTIX_RERANKER_URL</code>, <code>VANTIX_RERANKER_API_KEY</code></td><td>a small cross encoder, empty, empty</td><td>Model and endpoint</td></tr>
<tr><td><code>VANTIX_RERANKER_TOP_N</code></td><td><code>50</code></td><td>Number of fused candidates reranked</td></tr>
<tr><td colspan="3"><b>Generation</b></td></tr>
<tr><td><code>VANTIX_LLM_PROVIDER</code></td><td><code>extractive</code></td><td><code>extractive</code>, <code>openai_compatible</code> or <code>anthropic</code></td></tr>
<tr><td><code>VANTIX_LLM_MODEL</code>, <code>VANTIX_LLM_BASE_URL</code>, <code>VANTIX_LLM_API_KEY</code></td><td>provider defaults</td><td>Model and endpoint</td></tr>
<tr><td><code>VANTIX_LLM_TEMPERATURE</code>, <code>VANTIX_LLM_MAX_OUTPUT_TOKENS</code>, <code>VANTIX_LLM_TIMEOUT_SECONDS</code></td><td><code>0.1</code>, <code>700</code>, <code>60</code></td><td>Generation controls</td></tr>
<tr><td><code>VANTIX_RAG_CONTEXT_TOKEN_BUDGET</code></td><td><code>3000</code></td><td>Maximum tokens of sources per prompt</td></tr>
<tr><td><code>VANTIX_RAG_MIN_DENSE_SIMILARITY</code></td><td><code>0.2</code></td><td>Dense relevance gate for abstention; calibrate per embedding model</td></tr>
<tr><td><code>VANTIX_RAG_MIN_LEXICAL_COVERAGE</code></td><td><code>0.3</code></td><td>Share of query terms a passage must contain to pass the lexical gate</td></tr>
<tr><td colspan="3"><b>Recommender</b></td></tr>
<tr><td><code>VANTIX_REC_ALS_FACTORS</code>, <code>VANTIX_REC_ALS_REGULARIZATION</code>, <code>VANTIX_REC_ALS_ITERATIONS</code>, <code>VANTIX_REC_ALS_ALPHA</code></td><td><code>64</code>, <code>0.05</code>, <code>15</code>, <code>20</code></td><td>Implicit ALS hyperparameters</td></tr>
<tr><td><code>VANTIX_REC_PPR_DAMPING</code></td><td><code>0.85</code></td><td>Personalized PageRank damping</td></tr>
<tr><td><code>VANTIX_REC_INTERACTION_HALF_LIFE_DAYS</code></td><td><code>30</code></td><td>Decay of interaction strength for profiles and ALS</td></tr>
<tr><td><code>VANTIX_REC_POPULARITY_HALF_LIFE_DAYS</code></td><td><code>7</code></td><td>Decay of popularity and activity</td></tr>
<tr><td><code>VANTIX_REC_PROFILE_MAX_INTERACTIONS</code></td><td><code>200</code></td><td>Recent interactions used for a user's real time state</td></tr>
<tr><td><code>VANTIX_REC_CANDIDATE_POOL</code></td><td><code>300</code></td><td>Candidates per source</td></tr>
<tr><td><code>VANTIX_REC_TRAINING_INTERVAL_SECONDS</code></td><td><code>3600</code></td><td>Background retraining interval; 0 disables the scheduler</td></tr>
<tr><td><code>VANTIX_REC_WEIGHTS_DOCUMENTS</code></td><td><code>content:0.30,collaborative:0.25,graph:0.20,popularity:0.10,context:0.15</code></td><td>Signal weights for documents</td></tr>
<tr><td><code>VANTIX_REC_WEIGHTS_TASKS</code></td><td><code>content:0.25,collaborative:0.15,graph:0.20,popularity:0.05,context:0.25,urgency:0.10</code></td><td>Signal weights for tasks</td></tr>
<tr><td><code>VANTIX_REC_WEIGHTS_SIMILAR</code></td><td><code>context:0.50,collaborative:0.25,graph:0.25</code></td><td>Signal weights for similar items</td></tr>
<tr><td><code>VANTIX_REC_WEIGHTS_EXPERTS</code></td><td><code>content:0.35,graph:0.30,evidence:0.25,activity:0.10</code></td><td>Signal weights for experts</td></tr>
<tr><td><code>VANTIX_REC_MMR_LAMBDA</code></td><td><code>0.8</code></td><td>Diversity balance for recommendations</td></tr>
<tr><td colspan="3"><b>Workers</b></td></tr>
<tr><td><code>VANTIX_MODEL_REFRESH_SECONDS</code></td><td><code>30</code></td><td>How often each process checks the model directory for a newer model to hot reload</td></tr>
<tr><td><code>VANTIX_WORKER_HEARTBEAT_FILE</code></td><td><code>/tmp/vantix_worker_heartbeat</code></td><td>Heartbeat written by <code>vantix worker</code> and checked by <code>vantix health</code></td></tr>
<tr><td><code>VANTIX_INGESTION_WORKERS</code></td><td><code>2</code></td><td>Background ingestion threads</td></tr>
<tr><td><code>VANTIX_INGESTION_QUEUE_SIZE</code></td><td><code>10000</code></td><td>Maximum queued jobs before the API returns 503</td></tr>
</table>

## Evaluation and tuning

Vantix ships with an offline evaluation harness so that every change to models, weights or thresholds can be measured before it reaches users.

**Retrieval.** Prepare a JSON Lines file of labelled queries, one per line, with `query`, `relevant_item_ids` and an optional `principal`, then run:

```bash
vantix evaluate retrieval path/to/queries.jsonl workspace_acme
```

The report includes precision, recall, nDCG and hit rate at 10, and mean reciprocal rank. Collect labelled queries from search logs where users clicked a result, and from questions your support team answers repeatedly.

**Recommendation.** The recommender evaluation performs a strict temporal split: interactions before the 80th percentile timestamp are copied into an isolated in memory sandbox, a model is trained on them, and each user's later interactions with items they had not seen before become the ground truth. No information from after the cutoff can leak into training.

```bash
vantix evaluate recommender workspace_acme
```

The report includes precision, recall, nDCG, hit rate, mean reciprocal rank and catalogue coverage.

**Tuning workflow.**

1. Establish a baseline with both evaluations on a snapshot of production data.
2. Change one thing at a time: embedding model, reranker, fusion weights, signal weights or half lives.
3. Recalibrate `VANTIX_RAG_MIN_DENSE_SIMILARITY` whenever the embedding model changes: plot the dense score of labelled relevant and irrelevant passages and choose the threshold that keeps almost all relevant passages.
4. Validate online with feedback: compare recommendation click through, dismissal rate and the share of grounded answers before and after a change.

## Security and privacy

* **Tenant isolation.** The tenant is derived from the API key and applied to every query, index filter, cache key and model. Tests verify that one tenant can never read another tenant's data.
* **Access control everywhere.** ACL filters run inside Qdrant and OpenSearch queries and are rechecked in memory for every chunk. Recommendation explanations only reference items the principal can see.
* **Authenticity of changes.** Webhooks are signed with HMAC SHA256, compared in constant time, and protected against replay with a timestamp window.
* **Prompt injection hardening.** Sources are delimited, numbered and explicitly declared to be data; the model is instructed to ignore instructions inside sources; citations are validated after generation.
* **Right to erasure.** An `item.deleted` event or `DELETE /v1/items/{item_id}` removes all of the item's chunks from every index and clears its stored embedding immediately. `DELETE /v1/users/{user_id}/interactions` erases a user's entire interaction history and schedules retraining, so the user disappears from collaborative and graph signals at the next training run.
* **Data minimisation.** Only the fields the host sends are stored. Do not send fields that are not needed for search or recommendation.
* **Secrets.** API keys, webhook secrets and provider keys are read from the environment only. Never commit them; use a secrets manager.
* **Network posture.** Keep Vantix on a private network reachable only by the host backend. Expose `/v1/metrics` only to the monitoring system.

## Operations

**Probes.** Use `/v1/health` for liveness and `/v1/ready` for readiness; readiness reports the state of the database, vector index and lexical index individually and returns 503 when any is unavailable.

**Metrics.** `/v1/metrics` exposes Prometheus metrics, including:

<table>
<tr><th>Metric</th><th>Description</th></tr>
<tr><td><code>vantix_http_requests_total</code></td><td>Requests by method, route and status</td></tr>
<tr><td><code>vantix_http_request_seconds</code></td><td>Request latency histogram by route</td></tr>
<tr><td><code>vantix_stage_seconds</code></td><td>Latency of internal stages: recall, rerank, embedding, generation and each recommendation kind</td></tr>
<tr><td><code>vantix_items_indexed_total</code>, <code>vantix_chunks_indexed_total</code></td><td>Indexing throughput per tenant</td></tr>
<tr><td><code>vantix_events_total</code></td><td>Events by type and outcome (accepted or duplicate)</td></tr>
<tr><td><code>vantix_ingestion_queue_depth</code></td><td>Pending ingestion jobs</td></tr>
<tr><td><code>vantix_cache_hits_total</code>, <code>vantix_cache_misses_total</code></td><td>Retrieval cache effectiveness</td></tr>
<tr><td><code>vantix_rag_abstentions_total</code></td><td>Questions answered with an abstention</td></tr>
<tr><td><code>vantix_model_trainings_total</code></td><td>Training runs by tenant and outcome</td></tr>
</table>

Recommended alerts: readiness failing, error rate above 1 percent on any route, p95 latency of `/v1/ask` or `/v1/assist` above your target, queue depth growing for more than ten minutes, and any failed training run.

**Logs.** Logs are JSON lines with timestamp, level, logger, message, request identifier and tenant identifier, ready for any log pipeline.

**Rebuilding indexes.** Indexes can be rebuilt from the system of record at any time. Use `vantix reindex TENANT` (or `POST /v1/admin/reindex`) to push stored vectors back into the indexes, for example after restoring an index from scratch.

**Changing the embedding model.** Point `VANTIX_QDRANT_COLLECTION` and `VANTIX_OPENSEARCH_INDEX` at new names, set the new model and dimension, then run `vantix reindex TENANT full` (or `POST /v1/admin/reindex?full=true`) to rechunk and re embed every item into the new indexes. Switch traffic once the rebuild completes and delete the old collection and index afterwards. This gives a zero downtime migration.

**Backup and restore.** Back up PostgreSQL with your standard tooling; it contains everything required to restore service. Qdrant snapshots and OpenSearch snapshots shorten recovery time but are optional, because both can be rebuilt from PostgreSQL. Back up `VANTIX_MODEL_DIR` or simply retrain after restore.

## Continuous integration and delivery

Every change to Vantix passes through an automated pipeline that tests it, scans it, measures its ranking quality, builds a signed image exactly once, and promotes that same image through staging to production. Nothing reaches production that has not passed every gate, and every deployment can roll itself back.

```mermaid
flowchart LR
    subgraph PR["PULL REQUEST AND PUSH: CI"]
        direction TB
        STATIC["Static checks<br/>ruff, mypy, bandit,<br/>lockfile and SHA pin checks"]
        DOCS["README policy<br/>and diagram rendering"]
        TESTS["Tests on Python<br/>3.11, 3.12 and 3.13<br/>90 percent branch coverage"]
        INTEG["Integration tests<br/>PostgreSQL, Qdrant,<br/>OpenSearch services"]
        QUALITY["Model quality gate<br/>retrieval and<br/>recommendation floors"]
        IMAGE["Container build,<br/>Trivy scan, hardened<br/>run and smoke test"]
        CHART["Helm lint and<br/>kubeconform schemas"]
        SEC["Security workflows<br/>CodeQL, dependency review,<br/>pip audit, Gitleaks, Trivy"]
        GATE{"CI passed<br/>required check"}
        STATIC ==> GATE
        DOCS ==> GATE
        TESTS ==> GATE
        INTEG ==> GATE
        QUALITY ==> GATE
        IMAGE ==> GATE
        CHART ==> GATE
        SEC ==> GATE
    end

    subgraph MAIN["MERGE TO MAIN: CD"]
        direction TB
        BUILD["Build once<br/>amd64 and arm64"]
        SIGN["Push to GHCR,<br/>SBOM, SLSA provenance,<br/>cosign keyless signature"]
        STAGE["Deploy to staging<br/>verify signature, migrate,<br/>atomic rollout, helm test"]
        STAGESMOKE{"Staging smoke test"}
        BUILD ==> SIGN ==> STAGE ==> STAGESMOKE
    end

    subgraph TAG["VERSION TAG: RELEASE"]
        direction TB
        VERIFY["Version agreement<br/>and full CI rerun"]
        PROMOTE["Promote the tested<br/>staging digest"]
        PUBLISH["GitHub release:<br/>wheel, sdist, SBOM,<br/>Helm chart, checksums"]
        APPROVE{"Production approval<br/>by required reviewers"}
        PROD["Deploy to production<br/>signature verified,<br/>atomic, helm test"]
        PRODSMOKE{"Production smoke test"}
        VERIFY ==> PROMOTE ==> PUBLISH ==> APPROVE ==> PROD ==> PRODSMOKE
    end

    ROLLBACK["Automatic helm rollback<br/>to the last good revision"]

    GATE ==> BUILD
    STAGESMOKE ==> VERIFY
    STAGESMOKE ==> ROLLBACK
    PRODSMOKE ==> ROLLBACK

    classDef default fill:none
    style PR fill:none
    style MAIN fill:none
    style TAG fill:none
```

### Workflows

<table>
<tr><th>Workflow</th><th>Trigger</th><th>What it does</th></tr>
<tr><td><code>ci.yml</code></td><td>Every pull request, push to main and merge queue; reused by releases</td><td>Lint, formatting, types, Bandit, lockfile freshness, Action SHA pinning, README policy with diagram rendering in light and dark themes, tests on three Python versions with a 90 percent branch coverage gate, the full suite against real PostgreSQL, Qdrant and OpenSearch service containers, the model quality gate, a container build with hadolint, a Trivy image scan, a hardened container run (read only root filesystem, no capabilities, non root) followed by the smoke test and worker checks, Helm lint and kubeconform validation of every values file, and actionlint. A final <code>CI passed</code> job aggregates everything for branch protection</td></tr>
<tr><td><code>security.yml</code></td><td>Pull requests, main and weekly</td><td>Dependency review on pull requests, pip audit of both lockfiles, Bandit with SARIF upload, Gitleaks over the full history, Trivy filesystem, secret and misconfiguration scanning, and OpenSSF Scorecard</td></tr>
<tr><td><code>codeql.yml</code></td><td>Pull requests, main and weekly</td><td>CodeQL security and quality analysis of the Python code, the workflows and the diagram validator</td></tr>
<tr><td><code>cd.yml</code></td><td>CI succeeding on a push to main</td><td>Builds the image once for linux/amd64 and linux/arm64, pushes it to GHCR with an SBOM and SLSA provenance, signs it keylessly with Sigstore cosign, attests provenance, then deploys it to staging</td></tr>
<tr><td><code>release.yml</code></td><td>A version tag such as <code>v1.2.0</code></td><td>Checks that the tag, package, module and chart versions agree, reruns CI on the tagged commit, promotes the already tested image by digest (building only if CD never built that commit), publishes Python distributions, the image SBOM, the Helm chart and checksums as a GitHub release, then deploys to production; pre release tags such as <code>v1.2.0rc1</code> deploy to staging instead</td></tr>
<tr><td><code>deploy.yml</code></td><td>Called by CD and release, or run by hand</td><td>Resolves the image to a digest, verifies its signature came from this repository's pipelines, records the current revision, runs an atomic Helm upgrade (migrations first, waits for healthy pods), runs <code>helm test</code> and the live smoke test, and rolls back automatically if anything fails</td></tr>
<tr><td><code>nightly.yml</code></td><td>Daily</td><td>Runs the suite against the newest PostgreSQL, Qdrant and OpenSearch releases, a concurrent load test against p95 latency and error budgets, and a full quality report; any failure opens or updates a tracking issue</td></tr>
</table>

### Quality gates

* **Correctness.** Unit, service, contract and API tests; the same contract tests run against every vector and lexical backend, so access control and tenant isolation are proven on Qdrant and OpenSearch as well as in memory.
* **Coverage.** Branch coverage must stay at or above 90 percent.
* **Ranking quality.** `scripts/quality_gate.py` fails the build if retrieval nDCG, mean reciprocal rank, recall or hit rate, or recommender hit rate or catalogue coverage fall below the floors in `scripts/quality_thresholds.json`. The report appears in the job summary.
* **Performance.** The nightly load test fails when p95 latency exceeds the budget or any request errors.
* **Supply chain.** Hash verified lockfiles, Actions pinned to commit SHAs, dependency review, vulnerability audits, image scanning, signed images with SBOM and provenance, and signature verification before every deployment.
* **Documentation.** `scripts/check_readme.py` and the Mermaid validator keep this README free of dashes and tables of contents, and keep every diagram free of fills and backgrounds.

### One time GitHub configuration

1. **Branch protection on main.** Require pull requests, at least one review from code owners, and the status checks `CI passed`, `Analyze (python)` and the security jobs. Enable the merge queue if you use it.
2. **Environments.** Create `staging` and `production` under Settings, Environments. Give `production` required reviewers and restrict it to tags matching `v*`.
3. **Environment variables.** For each environment set `KUBE_NAMESPACE`, `VANTIX_URL` (the public base URL) and optionally `HELM_RELEASE`.
4. **Environment secrets.** For each environment set `KUBE_CONFIG` (a base64 encoded kubeconfig for a service account limited to the namespace), `VANTIX_SMOKE_API_KEY` (a key for a dedicated smoke test tenant) and `VANTIX_SMOKE_WEBHOOK_SECRET`.
5. **Packages.** The workflows publish to `ghcr.io/OWNER/vantix` and `ghcr.io/OWNER/charts` with the built in token. Grant the cluster pull access, or make the package internal to your organisation.
6. **Code owners.** Replace the placeholder team handles in `.github/CODEOWNERS`, and the image repository in `deploy/helm/vantix/values.yaml`.
7. **Security features.** Enable Dependabot alerts, secret scanning with push protection, and code scanning so that SARIF results appear under the Security tab.

### Releasing

1. Bump the version in `pyproject.toml`, `src/vantix/__init__.py` and `deploy/helm/vantix/Chart.yaml`, and add a `CHANGELOG.md` entry.
2. Merge to main. CD deploys the commit to staging automatically.
3. Tag the merged commit and push the tag:

   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```

4. Approve the production deployment when GitHub asks. The release promotes the exact image that ran in staging.

To roll back by hand, run the Deploy workflow from the Actions tab with the previous version's image reference; it is verified and deployed exactly like a release.

### Running the pipeline locally

Every CI check has a Make target, so problems surface before a push. `make help` lists them all.

```bash
make dev        # locked development dependencies and an editable install
make ci         # lint, types, security, README policy, tests with coverage, quality gate
make docker     # production image
make lock       # regenerate lockfiles after changing dependencies (requires uv)
make pin        # repin GitHub Actions to the newest commit SHA of each major version
```

To run the integration suite against real backends, start them with `docker compose up` and set `VANTIX_TEST_DATABASE_URL`, `VANTIX_TEST_QDRANT_URL` and `VANTIX_TEST_OPENSEARCH_URL` before `make integration`. Each test gets its own PostgreSQL schema, Qdrant collection and OpenSearch index, so runs never interfere. The precommit configuration in the repository root runs the fast checks on every commit.

`python examples/generate_sample_data.py` regenerates the deterministic sample workspace, and `python examples/host_integration.py` exercises a running server exactly as a host platform would.

## Extending Vantix

Every provider sits behind a small abstract interface. To add one, implement the interface and register it in the matching factory.

<table>
<tr><th>Extension</th><th>Interface</th><th>Factory</th></tr>
<tr><td>Embedding provider</td><td><code>vantix.embeddings.base.Embedder</code></td><td><code>vantix.embeddings.build_embedder</code></td></tr>
<tr><td>Vector index</td><td><code>vantix.index.base.VectorIndex</code></td><td><code>vantix.index.build_vector_index</code></td></tr>
<tr><td>Lexical index</td><td><code>vantix.index.base.LexicalIndex</code></td><td><code>vantix.index.build_lexical_index</code></td></tr>
<tr><td>Reranker</td><td><code>vantix.retrieval.rerankers.Reranker</code></td><td><code>vantix.retrieval.rerankers.build_reranker</code></td></tr>
<tr><td>Language model</td><td><code>vantix.rag.llm.LLM</code></td><td><code>vantix.rag.llm.build_llm</code></td></tr>
<tr><td>Recommendation signal</td><td>Add a scorer in <code>RecommendationEngine</code></td><td>Give it a weight in the <code>VANTIX_REC_WEIGHTS_*</code> variables</td></tr>
</table>

Index adapters must apply `SearchFilter` inside the backend query and return only chunks for which `SearchFilter.matches` is true, so that access control is preserved.

## Troubleshooting

<table>
<tr><th>Symptom</th><th>Likely cause and fix</th></tr>
<tr><td>Every request returns 401</td><td>The bearer key is not listed in <code>VANTIX_API_KEYS</code>, or the header is not in the form <code>Bearer key</code></td></tr>
<tr><td>Events return 401 but other calls work</td><td>Signature mismatch: sign the exact raw body bytes you send, use the current Unix time, and check the shared secret</td></tr>
<tr><td>Search returns nothing for a user</td><td>The principal lacks the item's project or group, or the items are still queued; check <code>GET /v1/jobs/{job_id}</code> and <code>GET /v1/admin/stats</code></td></tr>
<tr><td>Answers abstain too often</td><td>The dense gate is too strict for the embedding model; recalibrate <code>VANTIX_RAG_MIN_DENSE_SIMILARITY</code> with the evaluation harness</td></tr>
<tr><td>Recommendations are not personalised</td><td>No interactions are being sent, or no training run has happened yet; check <code>model_trained_at</code> and send <code>interaction.recorded</code> events</td></tr>
<tr><td>Permission change not reflected</td><td>Send an <code>item.upserted</code> event with the updated <code>acl</code>; results update as soon as the job completes</td></tr>
<tr><td>Startup fails with a dimension error</td><td><code>VANTIX_EMBEDDING_DIM</code> does not match the model or the existing Qdrant collection; follow the embedding model change procedure</td></tr>
<tr><td>CI fails on the lockfile check</td><td>Dependencies in <code>pyproject.toml</code> changed without regenerating the locks; run <code>make lock</code> and commit both files</td></tr>
<tr><td>CI fails the quality gate</td><td>A change lowered ranking quality; compare the job summary with the previous run, and only lower a floor in <code>scripts/quality_thresholds.json</code> with a written justification</td></tr>
<tr><td>Deploy fails at signature verification</td><td>The image was not built and signed by this repository's CD or release workflow, for example a manually pushed tag; deploy a digest produced by the pipeline</td></tr>
<tr><td>Deploy rolled back automatically</td><td>The rollout, <code>helm test</code> or the smoke test failed; the job log and summary show which step, and the previous revision is already serving traffic</td></tr>
<tr><td>Recommendations stop improving in Kubernetes</td><td>Check the worker pod: <code>vantix health</code> must pass and its log should list trained tenants; API pods hot reload models from the shared volume</td></tr>
<tr><td>Ingestion returns 503</td><td>The queue is full; slow the relay down, retry with backoff, or raise <code>VANTIX_INGESTION_WORKERS</code></td></tr>
</table>

## Licence

Copyright Phexara Ltd. All rights reserved. Proprietary and confidential. Unauthorised copying, distribution or use of this software, in whole or in part, is prohibited without the prior written permission of Phexara Ltd.
