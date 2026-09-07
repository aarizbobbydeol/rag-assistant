/**
 * TypeScript mirrors of the pydantic models in `app/models.py`.
 *
 * Field names are the snake_case wire names on purpose: a response body is cast
 * straight to these types with no adapter layer, so any drift between backend
 * and frontend shows up as a compile error rather than an `undefined` at
 * runtime.
 */

export type RetrievalMode = 'dense' | 'lexical' | 'hybrid';

export type TurnRole = 'user' | 'assistant';

export interface Chunk {
  chunk_id: string;
  doc_id: string;
  source: string;
  title: string;
  text: string;
  ordinal: number;
  start_char: number;
  end_char: number;
  page: number | null;
  token_count: number;
  metadata: Record<string, unknown>;
}

export interface ScoredChunk {
  chunk: Chunk;
  score: number;
  dense_score: number | null;
  lexical_score: number | null;
  rerank_score: number | null;
  rank: number;
  retriever: string;
}

export interface Citation {
  marker: number;
  chunk_id: string;
  doc_id: string;
  source: string;
  title: string;
  page: number | null;
  quote: string;
  score: number;
}

export interface SentenceSupport {
  sentence: string;
  supported: boolean;
  support_score: number;
  cited_markers: number[];
}

export interface Groundedness {
  score: number;
  supported_sentences: number;
  total_sentences: number;
  sentences: SentenceSupport[];
  unsupported: string[];
  invalid_citations: number[];
  uncited_sentences: number;
  abstained: boolean;
  reason: string | null;
}

export interface Usage {
  provider: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
}

export interface AnswerResult {
  question: string;
  standalone_question: string;
  answer: string;
  citations: Citation[];
  contexts: ScoredChunk[];
  groundedness: Groundedness;
  usage: Usage;
  latency_ms: Record<string, number>;
  session_id: string | null;
  trace_id: string;
}

export interface Turn {
  role: TurnRole;
  content: string;
  trace_id: string | null;
}

export interface ChatRequest {
  question: string;
  session_id?: string | null;
  top_k?: number | null;
  mode?: RetrievalMode | null;
  rerank?: boolean | null;
  strict?: boolean | null;
  include_contexts?: boolean;
}

export interface SearchRequest {
  query: string;
  top_k?: number;
  mode?: RetrievalMode | null;
  rerank?: boolean | null;
}

export interface SearchResponse {
  query: string;
  results: ScoredChunk[];
  elapsed_ms: number;
}

export interface IngestedDoc {
  doc_id: string;
  source: string;
  title: string;
  chunks: number;
  characters: number;
}

export interface IngestResponse {
  documents: IngestedDoc[];
  total_documents: number;
  total_chunks: number;
  skipped: string[];
  elapsed_ms: number;
  index_size: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  index_size: number;
  documents: number;
  embedder: string;
  llm_provider: string;
  llm_model: string;
  llm_available: boolean;
  vector_store: string;
  reranker: string;
}

export interface SessionResponse {
  session_id: string;
  turns: Turn[];
}

export interface ErrorResponse {
  error: string;
  detail: string | null;
  trace_id: string | null;
}

/**
 * `AuthService.register` returns a `User` and `issue_token` returns
 * `(jwt, expires_in_s)`. The user record is declared in `app/auth.py` rather
 * than `app/models.py`, so only the field this UI actually renders is required;
 * everything else stays optional to survive backend additions.
 */
export interface User {
  email: string;
  id?: string | number;
  created_at?: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user?: User;
}

/** File suffixes accepted by `app.ingestion.loaders.SUPPORTED_SUFFIXES`. */
export const SUPPORTED_SUFFIXES = [
  '.pdf',
  '.txt',
  '.md',
  '.markdown',
  '.html',
  '.htm',
  '.json',
  '.csv',
  '.rst',
  '.log',
] as const;
