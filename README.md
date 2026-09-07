# RAG Knowledge Assistant

Ask questions about your own PDFs and internal documents; get answers with
inline `[n]` citations back to the exact passage, and a groundedness score that
says how much of the answer the sources actually support.

```
POST /api/chat  {"question": "How quickly must the on-call engineer acknowledge a SEV-1 page?"}

{
  "answer": "The primary on-call engineer must acknowledge a SEV-1 page within 5 minutes. [1]",
  "citations": [
    {"marker": 1, "title": "Incident Response Runbook", "page": null,
     "source": "data/corpus/incident-response-runbook.md",
     "quote": "The primary on-call engineer must acknowledge a SEV-1 page within 5 minutes."}
  ],
  "groundedness": {"score": 1.0, "supported_sentences": 1, "total_sentences": 1},
  "usage": {"total_tokens": 3172, "estimated_cost_usd": 0.0},
  "latency_ms": {"condense": 0.0, "retrieve": 11.0, "generate": 16.0, "verify": 14.0}
}
```

**Stack** — Python 3.11+ · FastAPI · React + TypeScript (Vite) · NumPy/FAISS,
pgvector or Qdrant · any OpenAI-compatible LLM · Docker · Prometheus

---

## What makes it more than a chatbot wrapper

| | |
|---|---|
| **Hybrid retrieval** | Dense vectors and BM25 run in parallel, fused with reciprocal rank fusion, then diversified with MMR. Lexical recall catches the exact identifiers ("SEV-1", "FIDO2") that embeddings blur together. |
| **Reranking** | A heuristic reranker ships by default; cross-encoder or LLM rerankers drop in through config. Measured at **+0.039 nDCG** in the ablation below. |
| **Real citations** | Every factual sentence carries a `[n]` marker resolved to a chunk id, source file, page number and supporting quote. Markers the model invents are stripped and the survivors renumbered. |
| **Hallucination guard** | Two pre-generation gates plus a per-sentence check of each claim against the chunk it cites. Refusing correctly is measured as its own metric. |
| **Evaluation harness** | A 66-question golden set with multi-hop and deliberately-unanswerable cases, retrieval and answer metrics, and a one-command 30-config ablation sweep. |
| **Conversation memory** | Follow-ups are condensed into standalone questions before retrieval, so "and what about SEV-2?" still finds the right document. |
| **Production shell** | JWT auth, structured JSON logs with request correlation, Prometheus metrics, token/cost accounting, healthcheck-ready multi-stage container. |

**It runs with no API key.** The default embedder is a self-contained hashing
embedder and the default answerer is extractive, so a clone-and-run gives you a
working assistant and reproducible eval numbers offline. Set `RAG_LLM_API_KEY` to
switch generation to a real model; nothing else changes.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

```bash
.venv/Scripts/python scripts/make_sample_corpus.py
```

```bash
.venv/Scripts/python scripts/smoke.py
```

```bash
.venv/Scripts/python -m uvicorn app.main:app --reload
```

Then open <http://localhost:8000/docs>. On Linux/macOS use `.venv/bin/python`.

Frontend (dev server proxies `/api` to port 8000, so there is no CORS story):

```bash
cd web && npm install && npm run dev
```

Docker:

```bash
cp .env.example .env && docker compose up --build
```

Optional server-side vector stores and monitoring:

```bash
docker compose --profile pgvector up --build
```

```bash
docker compose --profile qdrant up --build
```

```bash
docker compose --profile monitoring up
```

---

## Using it

```bash
curl -X POST localhost:8000/api/ingest/paths -H 'content-type: application/json' -d '{"paths": ["data/corpus"], "recursive": true}'
```

```bash
curl -X POST localhost:8000/api/ingest/upload -F 'files=@handbook.pdf'
```

```bash
curl -X POST localhost:8000/api/chat -H 'content-type: application/json' -d '{"question": "What is the password minimum length?", "session_id": "demo"}'
```

```bash
curl -X POST localhost:8000/api/search -H 'content-type: application/json' -d '{"query": "incident severity", "top_k": 5}'
```

| Route | Purpose |
|---|---|
| `POST /api/chat` | Answer a question with citations and a groundedness verdict |
| `POST /api/search` | Retrieve chunks without calling the LLM |
| `POST /api/ingest/upload` · `/api/ingest/paths` | Add documents |
| `GET/DELETE /api/ingest/documents[/{doc_id}]` | Inspect and prune the index |
| `GET/DELETE /api/sessions/{id}` | Replay or forget a conversation |
| `POST /api/auth/register` · `/api/auth/token` · `GET /api/auth/me` | Accounts and JWTs |
| `GET /api/admin/stats` · `/api/admin/config` | Index stats, redacted config |
| `GET /health` · `/live` · `/ready` · `/metrics` | Ops |

---

## How it works

```
  upload ──> loaders ──> splitters ──> embedder ──> vector store
                                          │         (numpy | faiss | pgvector | qdrant)
                                          └──────> BM25 index

  question ──> condense ──> ┌ dense ┐
                            │       ├─ RRF fuse ─> MMR ─> rerank ─> prompt ─> LLM
                            └ BM25  ┘                                          │
                                                                               v
                                          groundedness guard <── answer + [n] citations
```

1. **Load** — PDF (via `pypdf`, tracking page spans), Markdown, HTML, text, JSON, CSV.
2. **Chunk** — four strategies, sized in tokens, carrying exact character offsets
   and page numbers so a citation can point at a page.
3. **Embed & index** — vectors into the store, raw text into BM25, written
   together so recall never silently degrades.
4. **Retrieve** — dense + lexical, fused by RRF, diversified by MMR, reranked.
5. **Generate** — numbered context blocks and a system prompt that forbids
   outside knowledge, mandates markers, and specifies a refusal phrase.
6. **Verify** — resolve markers to chunks, drop invented ones, score every
   sentence against its cited chunk.

Full module contracts: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Evaluation

The golden set (`data/golden/qa.jsonl`) is pinned to **verbatim source text**,
not chunk ids — so the same questions stay valid when the chunking strategy
changes, which is what makes the ablation below meaningful.

```bash
python -m eval.run_eval
```

```bash
python -m eval.sweep --out docs/RESULTS.md
```

Metrics: `recall@k`, `precision@k`, `hit_rate@k`, `MRR`, `nDCG@k` for retrieval;
`citation_precision`, `citation_recall`, `groundedness`, `token_f1`, `rouge_l`
for answers; and for refusal behaviour `abstain_recall` and **`false_answer_rate`**
— the fraction of genuinely unanswerable questions the system answered anyway,
which is the number a hallucination guard lives or dies by.

### Measured results

66 questions over a 7-document corpus (6 Markdown + 1 multi-page PDF), default
configuration: recursive chunking at 700/120 tokens, hybrid retrieval, heuristic
reranking, `top_k=5`, hashing embedder, offline extractive answerer.

| Retrieval (58 answerable) | | Answer | | Refusal (8 unanswerable) | |
|---|---|---|---|---|---|
| recall@5 | **0.994** | citation precision | 0.802 | abstain recall | **1.000** |
| nDCG@5 | **0.924** | citation recall | 0.859 | false-answer rate | **0.000** |
| MRR | **0.915** | token F1 | 0.441 | over-refusal rate | **0.034** |
| hit rate@5 | **1.000** | ROUGE-L | 0.390 | | |

Retrieval is the strong part: the right passage is in the top 5 for every
question and ranked first or second almost always. Per-tag recall@5 is 1.000 for
single-hop, security, API and handbook questions, 0.958–0.974 for multi-hop and
PDF-sourced ones.

### The refusal number, and how it got there

`false_answer_rate` started at **0.500** — four of the eight unanswerable
questions were answered anyway, fluently and with a citation. They were not
retrieval failures. Retrieval found the right *document* every time; the
specific fact simply was not in it:

| Question | What came back instead |
|---|---|
| acknowledgement target for a **SEV-4** | the SEV-2 and SEV-3 targets |
| **annual maximum benefit** on the dental plan | the sentence naming the plan |
| **cost per month** of the Enterprise tier | its requests-per-minute limit |
| who is the **Chief Executive Officer** | the CFO and the CISO |

Threshold tuning could not fix this. Two of the four scored `query_term_coverage`
of **1.000**, above the median for questions that *are* answerable — because
coverage pools the vocabulary of all five retrieved chunks, so "annual" and
"maximum" can arrive from two unrelated passages and satisfy it between them.
Raising the threshold would have cost good answers long before it caught these.

What fixed it was a different question — not *"do the question's words appear
somewhere?"* but *"is each discriminative term positively attested in the
passage we are about to answer from?"* Three detectors, no LLM call
([`app/generation/anchors.py`](app/generation/anchors.py)):

- **identifier** — `SEV-4` is present in the corpus, inside the sentence
  "There is no SEV-4", so attestation is negation-scoped. Deprecation is not
  negation: "v1 was retired" still attests v1.
- **name phrase** — "Chief Executive Officer" is fully covered word-by-word by
  the real CFO and CISO titles; only the combination is invented, so matching is
  on adjacent stemmed pairs.
- **orphan span** — two *adjacent* question terms absent from the best-matching
  chunk but present elsewhere in the corpus, which is what "annual maximum"
  looks like in a passage that only names the plan.

Plus one post-generation rule: a question asking a price whose answer contains
no money is refused, which is what the Enterprise-tier answer was.

Ablated at three retrieval depths, the guard costs nothing:

| top_k | false-answer rate | over-refusal rate |
|---|---|---|
| 3 | 0.500 → **0.000** | 0.052 → 0.052 |
| 5 | 0.500 → **0.000** | 0.034 → 0.034 |
| 8 | 0.500 → **0.000** | 0.034 → 0.034 |

Not one additional answerable question is refused, at any depth. Each detector
fires on well under 5% of questions by design: high precision, deliberately low
recall, so the union refuses far more rarely than any single threshold could.

### Ablation: what actually moved the numbers

30 configurations — 6 chunking strategies × 5 retrieval strategies, same 66
questions. Full table in [`docs/RESULTS.md`](docs/RESULTS.md).

| Configuration | nDCG@5 | Read |
|---|---|---|
| dense only (`recursive/700`) | 0.783 | the hashing embedder alone is the weakest retriever |
| lexical only (BM25) | 0.871 | exact identifiers carry this corpus |
| hybrid (RRF) | 0.885 | fusion beats either retriever alone |
| **hybrid + rerank** | **0.924** | reranking is the single biggest win: **+0.039** |
| best overall (`recursive/1100/180`, rerank, k=8) | **0.947** | larger chunks suit short policy documents |
| `semantic/700` splitter | 0.618–0.769 | worst by a wide margin |

Three results worth stating plainly, because they are why you run a sweep rather
than assert a design:

1. **Reranking earned its place** — +0.039 nDCG and +0.026 citation precision at
   identical retrieval settings, on every chunking strategy tested.
2. **Lexical beat dense here**, inverting the usual "just use embeddings"
   assumption. A corpus dense with identifiers and numbers rewards BM25, which is
   exactly why the default is hybrid rather than either alone.
3. **The semantic splitter lost badly** — it fragmented 7 documents into 173
   chunks, and small chunks starve the reranker of context. A plausible-sounding
   strategy measurably underperformed a plain recursive split.

### The refusal numbers, honestly

With the offline extractive answerer, half the deliberately-unanswerable
questions still get an answer. That is not a tuning failure — it is a real
property of a lexical guard, and the golden set was written to expose it. Six of
the eight unanswerable questions use *only vocabulary that exists in the corpus*:
"what is the acknowledgement target for a SEV-4 incident?" has 100% query-term
coverage, because SEV levels and acknowledgement targets are both discussed at
length; only SEV-4 specifically is absent. No amount of word overlap distinguishes
that from an answerable question — it needs reading comprehension.

Two guards run before generation and catch the cases that *are* lexically
detectable, at a cost of only 3.4% over-refusal:

| Guard | Catches |
|---|---|
| `min_retrieval_score` | nothing in the corpus is close to the question |
| `min_query_coverage` | the question's subject appears nowhere in the retrieved text |

Both thresholds were chosen by sweeping them against `false_answer_rate` and
`over_refusal_rate` on the golden set, not by intuition. Closing the remaining
gap is what the real-LLM path plus `RAG_STRICT_GROUNDING=true` is for: a model
that reads "SEV-1, SEV-2, SEV-3" and notices SEV-4 is missing will refuse where
word overlap cannot.

Groundedness is 1.000 by construction with the extractive backend — it quotes
source sentences verbatim, so it cannot fabricate. That number only becomes
informative against a generative model, which is precisely why the guard exists.

---

## Configuration

Every setting is an environment variable with the `RAG_` prefix, or a line in
`.env`. See [`.env.example`](.env.example) for the annotated list.

| Variable | Default | Notes |
|---|---|---|
| `RAG_LLM_API_KEY` | *(empty)* | Empty = offline extractive answerer |
| `RAG_LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint: Groq, Together, vLLM, Ollama |
| `RAG_EMBEDDING_BACKEND` | `auto` | `hashing` · `sentence-transformers` · `openai` |
| `RAG_VECTOR_STORE` | `auto` | `numpy` · `faiss` · `pgvector` · `qdrant` |
| `RAG_RETRIEVAL_MODE` | `hybrid` | `dense` · `lexical` · `hybrid` |
| `RAG_SPLITTER` / `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `recursive` / `700` / `120` | Tokens |
| `RAG_RERANK_ENABLED` / `RAG_RERANK_BACKEND` | `true` / `auto` | `heuristic` · `cross-encoder` · `llm` |
| `RAG_STRICT_GROUNDING` | `false` | `true` replaces ungrounded answers with a refusal |
| `RAG_MIN_RETRIEVAL_SCORE` / `RAG_MIN_QUERY_COVERAGE` | `0.10` / `0.65` | The two pre-generation gates |
| `RAG_AUTH_REQUIRED` / `RAG_AUTH_SECRET` | `false` / dev value | **Set both in production** |

Optional extras: `pip install -e ".[faiss]"`, `".[local-models]"`, `".[openai]"`,
`".[pgvector]"`, `".[qdrant]"`.

---

## Development

```bash
make test
```

```bash
make lint
```

```bash
make sweep
```

**513 tests pass** (`pytest`, ~37s) and `ruff` is clean. Coverage includes
chunk-offset invariants, BM25 and RRF behaviour, vector-store round-trips,
citation renumbering, JWT tampering and `alg:none` rejection, retry/backoff
against a mocked HTTP transport, the grounding guard, and full HTTP integration
through `TestClient`. The React app typechecks clean and builds.

Verified on Windows 11, Python 3.13.14, Node 22.15. **Docker is not installed on
the machine this was built on**, so the `Dockerfile` and `docker-compose.yml` are
written but unbuilt — treat them as reviewed, not run. The pgvector and Qdrant
backends are likewise wired and unit-tested for their failure paths, but have not
been exercised against live servers.

---

## Design notes

**Why hand-rolled instead of LangChain?** The retrieval seams here — fusion
weights, MMR, the reranker, the groundedness check — are exactly the parts a
framework hides behind a chain object, and exactly the parts worth showing. Each
is ~100 lines of readable code with its own tests, and swapping in LangChain or
LlamaIndex at the `RagIndex` boundary would be mechanical.

**Why an offline default?** A portfolio project that needs someone else's API key
to demonstrate anything mostly demonstrates that it cannot be verified. The
hashing embedder and extractive answerer make every number in this README
reproducible on a laptop with no network.

**Why lexical evidence outweighs embeddings in the grounding check?** A
paraphrase can score high on cosine similarity while asserting a fact the source
never made — precisely the failure a citation check exists to catch. Cosine is a
useful tiebreaker; overlap of content words is the stronger evidence.

**Known limits.** The default hashing embedder is weaker than a trained sentence
encoder — install the `local-models` extra for a real one. `NumpyVectorStore`
does exact search, which is right up to ~10⁵ chunks and wrong beyond; that is
what the FAISS, pgvector and Qdrant backends are for. Sessions live in process
memory, so a multi-worker deployment needs Redis behind the `SessionStore`
interface.
