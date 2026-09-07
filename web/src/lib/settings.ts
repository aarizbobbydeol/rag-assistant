import type { RetrievalMode } from '../api/types';
import { loadJson, saveJson } from './storage';

/** The retrieval knobs the settings drawer exposes, passed through on /api/chat. */
export interface ChatSettings {
  topK: number;
  mode: RetrievalMode;
  rerank: boolean;
  /** Abstain instead of returning an ungrounded answer (`ChatRequest.strict`). */
  strict: boolean;
}

/** Mirrors the defaults in `app.config.Settings`. */
export const DEFAULT_SETTINGS: ChatSettings = {
  topK: 5,
  mode: 'hybrid',
  rerank: true,
  strict: false,
};

export const TOP_K_MIN = 1;
export const TOP_K_MAX = 50;

export const RETRIEVAL_MODES: readonly RetrievalMode[] = ['hybrid', 'dense', 'lexical'];

export const RETRIEVAL_MODE_HELP: Record<RetrievalMode, string> = {
  hybrid: 'Dense vectors and BM25, fused by reciprocal rank. Best default.',
  dense: 'Embedding similarity only. Good for paraphrased questions.',
  lexical: 'BM25 only. Good for exact names, codes and identifiers.',
};

const STORAGE_KEY = 'rag.settings.v1';

function isMode(value: unknown): value is RetrievalMode {
  return value === 'hybrid' || value === 'dense' || value === 'lexical';
}

/** Reads stored settings, repairing any field that has drifted or gone stale. */
export function loadSettings(): ChatSettings {
  const raw = loadJson<Partial<ChatSettings>>(STORAGE_KEY, {});
  const topK =
    typeof raw.topK === 'number' && Number.isFinite(raw.topK)
      ? Math.min(TOP_K_MAX, Math.max(TOP_K_MIN, Math.round(raw.topK)))
      : DEFAULT_SETTINGS.topK;
  return {
    topK,
    mode: isMode(raw.mode) ? raw.mode : DEFAULT_SETTINGS.mode,
    rerank: typeof raw.rerank === 'boolean' ? raw.rerank : DEFAULT_SETTINGS.rerank,
    strict: typeof raw.strict === 'boolean' ? raw.strict : DEFAULT_SETTINGS.strict,
  };
}

export function saveSettings(settings: ChatSettings): void {
  saveJson(STORAGE_KEY, settings);
}
