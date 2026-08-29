# Adaptive Agentic RAG V1 Design Specification

**Date:** 2026-08-29  
**Status:** Approved for implementation planning  
**Scope:** Standalone V1 system design

## 1. Purpose

Build a portable, production-oriented Adaptive Agentic RAG system that can ingest tenant-scoped documents, retrieve evidence through hybrid BM25 and dense search, select an appropriate execution strategy per query, and produce observable, evaluated, and guarded answers.

The system must avoid paying the cost of an agent loop for simple questions while still supporting multi-hop research and tool use for difficult questions. The initial release will not depend on an object-storage service.

## 2. Goals

V1 must provide:

- Document ingestion for local uploads and filesystem sources.
- Parsing, normalization, metadata extraction, ACL propagation, chunking, deduplication, and versioning.
- Both BM25 and dense retrieval over the same indexed chunks.
- Reciprocal-rank fusion and configurable reranking.
- Tenant-aware semantic caching with safe invalidation.
- A two-axis adaptive router:
  - Depth: `parametric`, `single_hop`, or `multi_hop`.
  - Tool: `parametric`, `vector`, `sql`, or `web`.
- A bounded multi-hop agent loop with tool selection, query rewriting, evidence checks, and synthesis.
- Guardrails for input injection, access control, SQL safety, web safety, PII, groundedness, and citations.
- Arize Phoenix-compatible OpenTelemetry traces, spans, and evaluation annotations.
- FastAPI query and ingestion APIs.
- PostgreSQL persistence, migrations, health checks, Docker Compose, configuration validation, and automated tests.
- Reproducible offline evaluation plus live trace-based evaluation.

## 3. Non-goals for V1

The following are deliberately deferred:

- S3, GCS, Azure Blob, or another object-storage service.
- Multi-region replication and disaster recovery automation.
- Fine-tuning a routing or generation model.
- A production web frontend.
- Autonomous write access to external systems.
- Unbounded web crawling.
- Training a custom reranker.

Uploaded files may use a temporary local mounted directory during processing. The durable V1 record is canonical extracted text, metadata, chunks, embeddings, and index state. The storage seam must allow an object-store adapter to be added later without changing ingestion or query contracts.

## 4. System boundary

```text
                 INGESTION TIME
Local upload ─┐
Filesystem ───┼→ parse → normalize → ACL/metadata → chunk
Connectors ───┘                                      ↓
                              canonical text + chunks + embeddings
                                      ↓
                              PostgreSQL + OpenSearch

                 QUERY TIME
User query → cache → adaptive router → direct tool or agent loop
                                      ↓
                           hybrid retrieval / SQL / web
                                      ↓
                         rerank → synthesize → guardrails → answer
                                      ↓
                              Phoenix trace + annotations
```

PostgreSQL is the control-plane store for tenants, documents, chunks, ingestion jobs, query records, cache metadata, and evaluation records. OpenSearch is the default retrieval index for BM25 and dense k-NN search. The retrieval interface must remain backend-agnostic.

## 5. Core architecture

### 5.1 Ingestion layer

The ingestion layer is an asynchronous data pipeline. It accepts a source, creates an ingestion job, and processes the source through these stages:

1. Validate source type, size, tenant, and permissions.
2. Read local upload or filesystem input into a temporary workspace.
3. Parse PDF, DOCX, HTML, Markdown, and plain text into a canonical document.
4. Normalize encoding, whitespace, headings, tables, page markers, and source links.
5. Extract document metadata and attach server-derived tenant and ACL fields.
6. Compute a content hash and compare it with the active document version.
7. Chunk by document structure with token and overlap limits.
8. Generate embeddings in batches with retry and rate-limit handling.
9. Write canonical document/chunk records transactionally.
10. Index BM25 text and dense vectors in OpenSearch.
11. Mark the job and document version as active only after indexing succeeds.

Every stage must be idempotent. A failed job must retain an error code, safe diagnostic message, retry count, and stage name. Reprocessing the same content hash must not create duplicate active versions.

The minimum canonical models are:

```text
Document:
  id, tenant_id, source_type, source_uri, filename, content_hash,
  version, title, canonical_text, metadata, acl, status, timestamps

Chunk:
  id, document_id, tenant_id, version, ordinal, text, heading_path,
  page_number, token_count, metadata, acl, embedding_status, index_status

IngestionJob:
  id, tenant_id, document_id, status, current_stage, attempts,
  error_code, error_message, timestamps
```

Deletion and replacement must remove or deactivate the corresponding chunks in both search paths. All index operations must be scoped by tenant and document version.

### 5.2 Hybrid retrieval layer

The retrieval layer exposes one stable contract:

```python
retrieve(query: RetrievalQuery) -> RetrievalResult
```

`RetrievalQuery` contains the query text, tenant scope, ACL filters, requested limits, optional document filters, and trace context. `RetrievalResult` contains ranked chunks, per-stage scores, source metadata, and retrieval diagnostics.

The default implementation runs:

1. BM25 retrieval.
2. Dense vector retrieval.
3. Tenant/ACL filtering before results are returned.
4. Reciprocal-rank fusion.
5. Optional cross-encoder or provider reranking.
6. A final context budget and duplicate-removal pass.

The system must record BM25 hits, dense hits, fused rank, reranker score, and final selected chunks so retrieval quality can be evaluated independently from answer quality.

### 5.3 Semantic cache

The semantic cache runs before routing and generation. It stores normalized query embeddings, tenant and ACL scope, answer text, citations, model/profile, source document versions, cost, latency, and expiry.

A cache hit is valid only when:

- Tenant and ACL scope match.
- Query similarity exceeds the configured threshold.
- The answer has not expired.
- All referenced document versions are still active.
- The requested response mode and model policy are compatible.

Cache keys must not expose raw sensitive queries in logs. Cache misses, hits, bypasses, invalidations, and stale entries must be observable. A document replacement or deletion invalidates entries referencing that document version.

### 5.4 Adaptive router

The router is the adaptive agentic control layer. It does not retrieve documents or generate the final answer. It produces a typed decision:

```python
RouteDecision(
    depth: Literal["parametric", "single_hop", "multi_hop"],
    tool: Literal["parametric", "vector", "sql", "web"],
    confidence: float,
    source: Literal["llm", "embedding", "policy", "escalated"],
    rationale: str | None,
)
```

Routing behavior:

- `parametric` depth calls the parametric tool without retrieval.
- `single_hop` calls exactly one selected tool, subject to policy.
- `multi_hop` enters the bounded agent loop.
- `vector` uses the hybrid BM25+dense retriever.
- `sql` uses read-only, validated SQL over an approved schema.
- `web` uses an allowlisted and sanitized web fetcher.
- Low-confidence decisions escalate toward `multi_hop` according to policy and budget.
- The router cannot modify tenant scope, ACLs, budgets, or safety policy.

The primary router may be a cheap structured-output LLM. An embedding k-nearest-neighbor router is a swappable alternative. Both must be evaluated against the same labeled route dataset.

### 5.5 Multi-hop agent loop

The multi-hop loop is a bounded state machine, not an unconstrained autonomous process:

```text
plan → select tool → execute tool → observe evidence
  ↑                                  ↓
  └──── insufficient evidence ───────┘
                    ↓
                 synthesize
```

Each iteration may:

- Rewrite or decompose the query.
- Select vector, SQL, web, or parametric tools.
- Store a structured observation and citations.
- Decide whether evidence is sufficient.
- Finish with a grounded answer or refuse.

Every model and tool call charges a per-query budget. The loop stops on answer completion, maximum steps, maximum tokens, maximum cost, timeout, repeated identical action, or unrecoverable tool failure. Tool outputs are observations, not instructions.

### 5.6 Tool contracts

All tools implement:

```python
run(context: ToolContext) -> ToolResult
```

`ToolContext` contains the authenticated tenant/ACL context, query, budget, trace identifiers, and approved policy. `ToolResult` contains status, text or rows, citations, usage, latency, error code, and diagnostics.

Required tools:

- `ParametricTool`: answers without external retrieval and must be explicit about uncertainty.
- `VectorTool`: calls the hybrid retriever and returns chunk-level citations.
- `SqlTool`: generates SQL, validates it, injects server-side tenant scope, enforces limits/timeouts, and executes read-only queries.
- `WebTool`: fetches only approved domains, sanitizes content, applies timeouts, and labels freshness and source URLs.

### 5.7 Answer synthesis

The synthesizer receives only approved evidence and structured tool results. It must produce:

```text
answer text
citations
groundedness status
unsupported-claim findings
refusal status
usage and latency
```

It must not silently present unsupported claims as facts. Conflicting evidence must be surfaced or resolved using an explicit source policy.

## 6. Guardrails

Guardrails run at input, tool, evidence, and output boundaries.

### Input

- Validate request size, tenant identity, query format, and rate limits.
- Detect prompt-injection attempts where applicable.
- Prevent user text from becoming SQL or policy instructions.

### Retrieval and tools

- Apply ACL filters server-side before BM25, dense, or SQL results reach synthesis.
- Treat documents and web content as untrusted data.
- Validate SQL through an AST and allow only one read-only `SELECT` statement.
- Enforce row limits, statement timeouts, approved schemas, and read-only credentials.
- Restrict web access to configured domains and sanitize returned markup/scripts.
- Enforce agent budgets, tool allowlists, timeouts, and retry caps.

### Output

- Check groundedness against returned evidence.
- Check citation presence and citation-to-claim coverage.
- Detect PII and configured sensitive data.
- Refuse or qualify answers when evidence is insufficient.
- Never disclose hidden prompts, credentials, ACL rules, or internal diagnostics.

Every guardrail decision must have a machine-readable result and a trace annotation.

## 7. Phoenix observability and evaluation

The application will emit OpenTelemetry traces that can be viewed in Arize Phoenix. Provider-specific instrumentation may be added, but critical spans must also be created explicitly so the system remains observable across model providers.

### 7.1 Required span hierarchy

```text
rag.request
├── cache.lookup
├── router.decide
├── retrieval.hybrid
│   ├── retrieval.bm25
│   ├── retrieval.dense
│   ├── retrieval.fusion
│   └── retrieval.rerank
├── agent.run
│   └── agent.step
│       ├── tool.sql / tool.vector / tool.web / tool.parametric
│       └── agent.evidence_check
├── answer.synthesize
├── guardrail.input / guardrail.tool / guardrail.output
└── evaluation.record
```

### 7.2 Required span attributes

Spans must include request ID, trace ID, tenant-safe scope identifier, route depth, selected tool, actual tools used, cache status, model profile, token counts, cost, latency, step count, result counts, failure codes, and guardrail statuses.

Raw prompts, retrieved documents, credentials, and sensitive tenant data must be redacted or disabled by configuration. A stable query hash may be recorded for correlation.

### 7.3 Phoenix annotations

The evaluation process must attach annotations to traces/spans for:

- Answer correctness.
- Faithfulness/groundedness.
- Context relevance.
- Citation completeness.
- BM25 recall and dense recall.
- Reranker quality.
- Depth-route correctness.
- Tool-selection correctness.
- SQL execution accuracy.
- Agent step efficiency.
- Safety and guardrail outcome.
- Cost and latency budget compliance.

Annotations must include evaluator name/version, score, explanation, dataset item ID, and whether the label is human, reference-based, or model-assisted. Evaluator failures must not break query serving.

## 8. Evaluation requirements

The evaluation harness must support:

- Golden route labels for parametric, single-hop, and multi-hop queries.
- Golden tool labels for vector, SQL, web, and parametric cases.
- Retrieval metrics such as recall@k, precision@k, MRR, nDCG, and reranker lift.
- Answer metrics such as correctness, groundedness, citation completeness, and refusal quality.
- SQL result-set equivalence and safety rejection tests.
- Agent trajectory metrics: steps, tool calls, retries, cost, latency, and termination reason.
- Routed versus always-agentic comparison.
- Confidence-threshold sweep to find an accuracy-preserving operating point.
- Bootstrap confidence intervals for accuracy, cost, and latency deltas.

The initial deterministic datasets should include:

- Factoid questions for parametric answers.
- Document questions for single-hop vector retrieval.
- Multi-hop questions requiring iterative evidence gathering.
- Tenant-scoped SQL questions.
- Freshness questions using a reproducible web fixture or cache.
- Adversarial guardrail and prompt-injection cases.

No production performance claim may be published until the live evaluation run has populated the metrics and trace annotations.

## 9. APIs

Minimum V1 endpoints:

```text
POST /v1/queries
POST /v1/documents
POST /v1/documents/{document_id}/reindex
DELETE /v1/documents/{document_id}
GET  /v1/ingestion/{job_id}
GET  /health/live
GET  /health/ready
GET  /metrics
```

`POST /v1/queries` must return an answer, citations, route summary, trace ID, usage, latency, cache status, and refusal/guardrail status. Streaming may be added behind the same query contract but is not required for the first implementation slice.

Authentication may begin with a signed development identity adapter, but production interfaces must receive tenant and ACL context from an authenticated boundary rather than from request body fields.

## 10. Persistence model

PostgreSQL migrations must define at least:

- `tenants`
- `documents`
- `document_versions`
- `chunks`
- `ingestion_jobs`
- `query_runs`
- `semantic_cache_entries`
- `evaluation_annotations`

Foreign keys, tenant indexes, active-version constraints, created/updated timestamps, and safe deletion semantics are required. Search-index IDs must be deterministic from tenant, document version, and chunk ID.

## 11. Configuration and deployment

Configuration must be environment-driven and validated at startup. It must cover database DSN, OpenSearch endpoint, embedding and generation providers, Phoenix endpoint, auth mode, limits, cache thresholds, web allowlist, SQL policy, logging level, and redaction mode.

The repository must include:

- `.env.example` containing names only, never real secrets.
- Dockerfile for the application.
- Docker Compose for application, PostgreSQL, OpenSearch, and Phoenix.
- Named persistent volumes for PostgreSQL, OpenSearch, and Phoenix data where supported.
- Startup migrations and readiness checks.
- A documented one-command local startup path.
- A CI workflow for formatting, type checks, unit tests, integration tests, and image build.

The application must run without the original sibling `Production RAG` repository. External providers are configured through adapters and may be replaced by deterministic test doubles.

## 12. Testing and acceptance criteria

V1 is complete when:

1. A clean machine can start the stack from the repository instructions.
2. A document can be ingested, indexed in BM25 and dense search, and queried through the API.
3. Duplicate content does not create duplicate active versions.
4. Reindexing and deletion update both search paths.
5. Cross-tenant retrieval and SQL access are rejected by automated tests.
6. Parametric and single-hop queries avoid the agent loop.
7. Multi-hop queries use the bounded agent loop and terminate under all configured budgets.
8. SQL, vector, web, and parametric tools satisfy their contracts independently.
9. Semantic-cache hits are tenant-safe and invalidated by document version changes.
10. Every query produces Phoenix-compatible spans with required attributes.
11. Evaluation traces receive the required annotations without affecting serving.
12. Guardrails reject or qualify unsupported, unsafe, injected, or unauthorized requests.
13. CI passes without live provider credentials.
14. A live evaluation run can produce routed-versus-always-agentic cost, latency, routing, retrieval, and answer metrics.

## 13. Future extension seams

The following interfaces must be kept replaceable:

- `DocumentSource` for future object storage and SaaS connectors.
- `Parser` for additional file types.
- `EmbeddingProvider` and `Generator` for model changes.
- `HybridRetriever` for OpenSearch, PostgreSQL, or another backend.
- `Reranker` for cross-encoder or hosted rerankers.
- `Router` for LLM, embedding, policy, or fine-tuned classifiers.
- `CacheStore` for a future distributed cache.
- `TraceExporter` and `EvaluationAnnotator` for Phoenix or another backend.
