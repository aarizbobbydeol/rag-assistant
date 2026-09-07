# Architecture & module contracts

The assistant is a straight pipeline with swappable parts. Every seam is an
abstract base class, every implementation registers itself in a factory, and the
factory reads `app.config.Settings`. That is what makes the ablation sweep in
`eval/sweep.py` possible: a config dict in, a metrics row out.

```
  upload ──> loaders ──> splitters ──> embedder ──> vector store
                                                    (numpy | faiss | pgvector | qdrant)
                                          │
                                          └──> BM25 index

  question ──> condense ──> ┌ dense  ┐
                            │        ├─ RRF fuse ─> rerank ─> prompt ─> LLM
                            └ BM25   ┘                                   │
                                                                         v
                                     groundedness guard <── answer + [n] citations
```

React SPA -> `/api/*` (JWT bearer) -> FastAPI -> pipeline.

## Layer map

| Package | Owns | Key entry points |
|---|---|---|
| `app.ingestion.loaders` | bytes/paths -> `Document` | `load_path`, `load_bytes`, `iter_paths` |
| `app.ingestion.chunking` | `Document` -> `list[Chunk]` | `get_splitter`, `Splitter.split` |
| `app.retrieval.embeddings` | text -> unit vectors | `get_embedder`, `Embedder` |
| `app.retrieval.vectorstore` | vector search | `get_vector_store`, `VectorStore` |
| `app.retrieval.bm25` | sparse lexical scoring | `BM25Index` |
| `app.retrieval.hybrid` | fusion + MMR | `reciprocal_rank_fusion`, `mmr_select` |
| `app.retrieval.rerank` | candidate reordering | `get_reranker`, `Reranker` |
| `app.retrieval.index` | the whole retrieval side | `RagIndex` |
| `app.generation.llm` | provider abstraction | `get_llm`, `LLMClient` |
| `app.generation.prompts` | prompt text | `build_answer_messages` |
| `app.generation.citations` | `[n]` -> `Citation` | `renumber_answer` |
| `app.generation.guardrails` | hallucination checks | `check_groundedness` |
| `app.memory` | conversation state | `SessionStore` |
| `app.auth` | users, passwords, JWT | `AuthService` |
| `app.pipeline` | wires it together | `RagPipeline` |
| `eval.*` | offline quality measurement | `run_eval`, `sweep` |
| `web/` | React + Vite SPA | `src/App.tsx` |

## Contracts

All types come from `app.models`. All helpers come from `app.utils`
(`tokenize`, `estimate_tokens`, `split_sentences`, `normalize_whitespace`,
`stable_id`, `truncate`, `jaccard`, `containment`).

### `app.ingestion.loaders`

```python
SUPPORTED_SUFFIXES: set[str]   # .pdf .txt .md .markdown .html .htm .json .csv .rst .log

def load_path(path: str | Path) -> Document
def load_bytes(data: bytes, filename: str) -> Document
def iter_paths(root: str | Path, recursive: bool = True) -> Iterator[Path]
```

- `doc_id = stable_id(resolved_source_path)`.
- PDFs go through `pypdf`; page boundaries land in
  `Document.metadata["page_spans"] = [[start_char, end_char, page_no], ...]`
  so chunking can attribute a page to every chunk.
- Raise `UnsupportedFileType` for unknown suffixes, `DocumentLoadError` when
  extraction yields no text.
- Text is passed through `normalize_whitespace`.

### `app.ingestion.chunking`

```python
class Splitter(ABC):
    name: str
    def __init__(self, chunk_size: int, chunk_overlap: int, min_chunk_tokens: int = 24)
    def split(self, doc: Document) -> list[Chunk]

def get_splitter(name, chunk_size, chunk_overlap, min_chunk_tokens=24) -> Splitter
SPLITTERS: dict[str, type[Splitter]]   # recursive, fixed, sentence, semantic
```

- Sizes are measured in **tokens** via `estimate_tokens`.
- `chunk_id = stable_id(doc_id, str(ordinal), text[:64])`; `ordinal` is 0-based
  and contiguous per document.
- `start_char` / `end_char` must index back into `doc.text` exactly.
- `page` comes from `doc.metadata["page_spans"]` (the page holding `start_char`)
  when present, else `None`.
- `recursive` splits on `["\n## ", "\n# ", "\n\n", "\n", ". ", " "]` in order.
- `fixed` is a sliding window. `sentence` packs whole sentences. `semantic`
  packs sentences and breaks when consecutive-sentence lexical similarity drops.
- Chunks under `min_chunk_tokens` merge into the previous chunk.

### `app.retrieval.embeddings`

```python
class Embedder(ABC):
    name: str
    dim: int
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray  # (n, dim) f32, L2-normalised
    def embed_query(self, text: str) -> np.ndarray                 # (dim,) f32, L2-normalised

def get_embedder(settings) -> Embedder
```

- `HashingEmbedder` is the default and has **no model dependency**: word
  unigrams + bigrams + char 4-grams hashed into `dim` buckets with a signed
  hash, weighted by sublinear tf, L2-normalised. Deterministic across processes
  (use `hashlib`, never `hash()`).
- `SentenceTransformerEmbedder` and `OpenAIEmbedder` are optional, chosen by
  `embedding_backend`; `auto` prefers sentence-transformers, then OpenAI (only
  when `llm_api_key` is set), then hashing.
- Empty input returns a zero vector of the right shape, never an exception.

### `app.retrieval.vectorstore`

```python
class VectorStore(ABC):
    name: str
    dim: int
    def add(self, ids, vectors: np.ndarray) -> None
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]  # cosine, desc
    def delete(self, ids) -> int
    def clear(self) -> None
    def save(self, path: Path) -> None
    def load(self, path: Path) -> None
    def __len__(self) -> int

def get_vector_store(settings, dim: int) -> VectorStore
```

- `NumpyVectorStore` (default): `(n, dim)` float32 matrix, exact cosine via one
  matmul. `FaissVectorStore` wraps `IndexFlatIP`.
- `PgVectorStore` (`psycopg` + pgvector) and `QdrantStore` (`qdrant-client`) are
  the server-side backends used by `docker-compose`. Both are lazily imported so
  a missing driver never breaks an import.
- Scores are cosine in `[-1, 1]`; re-adding an id replaces it. `save`/`load`
  round-trip (no-ops for the server-backed stores, which persist themselves).

### `app.retrieval.bm25`

```python
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75)
    def add(self, ids, texts) -> None
    def search(self, query: str, k: int) -> list[tuple[str, float]]
    def delete(self, ids) -> int
    def clear(self) -> None
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index"
    def __len__(self) -> int
```

Pure Python + `collections.Counter`. Raw BM25 scores; fusion normalises.

### `app.retrieval.hybrid`

```python
def reciprocal_rank_fusion(rankings, k=60, weights=None) -> list[tuple[str, float]]
def weighted_fusion(dense, lexical, dense_weight, lexical_weight) -> list[tuple[str, float]]
def min_max_normalise(scores) -> list[float]
def mmr_select(query_vec, candidate_vecs, candidate_scores, k, lambda_mult=0.7) -> list[int]
```

RRF is the default: `score = sum(w_i / (k + rank_i))`, ranks 1-based.
`mmr_select` returns greedy indices into the candidate arrays.

### `app.retrieval.rerank`

```python
class Reranker(ABC):
    name: str
    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]

def get_reranker(settings, llm=None) -> Reranker
```

`NoopReranker`, `HeuristicReranker` (default: query-term containment +
phrase proximity + position prior, blended with the fusion score),
`CrossEncoderReranker` (optional), `LLMReranker` (falls back to heuristic on any
failure). Rerankers set `rerank_score`, rewrite `score` and `rank`, and never
mutate the input list in place.

### `app.retrieval.index`

```python
class RagIndex:
    def __init__(self, settings, embedder, store, reranker=None)
    def add_documents(self, docs) -> int          # chunks added
    def add_chunks(self, chunks) -> int
    def search(self, query, top_k, mode, rerank, candidate_k=None) -> list[ScoredChunk]
    def delete_document(self, doc_id: str) -> int
    def clear(self) -> None
    def save(self) -> None
    def load(self) -> bool
    def stats(self) -> dict
    def __len__(self) -> int
```

### `app.generation.llm`

```python
class LLMClient(ABC):
    provider: str ; model: str ; available: bool
    def complete(self, messages, max_tokens=None, temperature=None) -> LLMResponse

def get_llm(settings) -> LLMClient
```

- `OpenAICompatibleLLM` talks to any `/chat/completions` endpoint over `httpx`
  (OpenAI, Together, Groq, vLLM, Ollama's shim). Retries with backoff on
  429/5xx/timeout, raises `LLMError` / `LLMUnavailable`, fills `Usage` from the
  response plus configured per-1M prices.
- `ExtractiveLLM` is the offline default: it picks the context sentences that
  best cover the question and emits them with `[n]` markers. It makes the system
  runnable, testable and evaluable with **no API key**, at zero reported cost.
- `auto` picks OpenAI-compatible when `llm_api_key` is set, else extractive.

### `app.generation.prompts`

```python
SYSTEM_PROMPT: str
def format_context(contexts, max_tokens) -> tuple[str, list[ScoredChunk]]
def build_answer_messages(question, contexts, history, max_context_tokens) -> list[LLMMessage]
def build_condense_messages(question, history) -> list[LLMMessage]
def build_rerank_messages(query, candidates) -> list[LLMMessage]
```

Context blocks render as `[n] title (p.X) :: source` + text, `n` starting at 1
and matching the returned context order. The system prompt forbids outside
knowledge, requires a `[n]` marker on every factual sentence, and mandates the
abstention phrase when context is insufficient.

### `app.generation.citations`

```python
def extract_markers(text: str) -> list[int]
def resolve_citations(answer, contexts) -> tuple[list[Citation], list[int]]
def renumber_answer(answer, contexts) -> tuple[str, list[Citation], list[int]]
def strip_markers(text: str) -> str
```

Handles `[1]`, `[1,2]`, `[1][2]`. `renumber_answer` drops invalid markers from
the prose and renumbers survivors to a dense 1..n sequence.

### `app.generation.guardrails`

```python
def sentence_support(sentence, chunk_texts, embedder=None) -> float
def check_groundedness(answer, contexts, citations, invalid_markers, settings, embedder=None) -> Groundedness
def should_abstain(top_score: float, settings) -> bool
```

Support blends lexical containment of the sentence's content words in the cited
chunks with embedding cosine. Sentences with no marker are checked against all
contexts and counted in `uncited_sentences`.

### `app.memory`

```python
class SessionStore:
    def __init__(self, max_sessions=1000, ttl_s=21600)
    def history(self, session_id, limit) -> list[Turn]
    def append(self, session_id, turn) -> None
    def reset(self, session_id) -> bool
    def purge_expired(self) -> int
```

LRU + TTL, thread-safe.

### `app.auth`

```python
class AuthService:
    def __init__(self, settings, db_path: Path)
    def register(self, email: str, password: str) -> User
    def authenticate(self, email: str, password: str) -> User
    def issue_token(self, user: User) -> tuple[str, int]    # (jwt, expires_in_s)
    def verify_token(self, token: str) -> User
```

- Passwords: `hashlib.pbkdf2_hmac("sha256", ..., 260_000)` with a per-user salt,
  compared with `hmac.compare_digest`. No plaintext ever stored or logged.
- Tokens: HS256 JWT written by hand (`hmac` + `hashlib` + base64url) so there is
  no crypto dependency to audit; claims `sub`, `email`, `iat`, `exp`, `jti`.
- Storage: stdlib `sqlite3` (`users` table), file path from settings.
- `RAG_AUTH_REQUIRED=false` (default in dev) leaves the API open; when true,
  every `/api` route except `/health`, `/auth/*` and `/metrics` needs a bearer.

## Non-negotiables

1. **Runs with zero API keys and zero external services.** Default embedder,
   store, reranker and LLM are local and deterministic. Optional backends import
   lazily inside their factory.
2. **No mutable global state** other than the singleton pipeline in `app.deps`.
3. **Type hints on every public function**; docstrings where behaviour is not
   obvious from the name.
4. **Determinism.** Same corpus + config + question -> same answer. Never
   `hash()`, set-iteration order, or unseeded RNG in a scoring path.
