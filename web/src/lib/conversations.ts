/**
 * Conversation state that lives in the browser.
 *
 * The backend `SessionStore` keeps the transcript keyed by `session_id`, but a
 * `Turn` only carries role and content — the citations, contexts and
 * groundedness verdict that make an answer inspectable are not persisted there.
 * So the client caches full answers per session and falls back to the server
 * transcript when the cache is cold (another device, cleared storage).
 */

import type { AnswerResult, Turn } from '../api/types';
import { loadJson, randomId, removeKey, saveJson } from './storage';

export interface SessionMeta {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
}

export interface UserMessage {
  id: string;
  role: 'user';
  content: string;
  createdAt: number;
}

export interface AssistantMessage {
  id: string;
  role: 'assistant';
  content: string;
  createdAt: number;
  /** Null when the answer came back from the server transcript, or failed. */
  result: AnswerResult | null;
  error: string | null;
}

export type ChatMessage = UserMessage | AssistantMessage;

const SESSIONS_KEY = 'rag.sessions.v1';
const MESSAGES_PREFIX = 'rag.messages.v1.';

/** Enough history to scroll through without risking the storage quota. */
const MAX_CACHED_MESSAGES = 40;

export const NEW_SESSION_TITLE = 'New conversation';

function messagesKey(sessionId: string): string {
  return `${MESSAGES_PREFIX}${sessionId}`;
}

export function newSession(): SessionMeta {
  const now = Date.now();
  return { id: randomId('session'), title: NEW_SESSION_TITLE, createdAt: now, updatedAt: now };
}

export function loadSessions(): SessionMeta[] {
  const stored = loadJson<SessionMeta[]>(SESSIONS_KEY, []);
  if (!Array.isArray(stored)) return [];
  return stored
    .filter(
      (item): item is SessionMeta =>
        typeof item === 'object' &&
        item !== null &&
        typeof item.id === 'string' &&
        typeof item.title === 'string',
    )
    .map((item) => ({
      id: item.id,
      title: item.title,
      createdAt: Number(item.createdAt) || Date.now(),
      updatedAt: Number(item.updatedAt) || Date.now(),
    }))
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

export function saveSessions(sessions: SessionMeta[]): void {
  saveJson(SESSIONS_KEY, sessions);
}

export function loadMessages(sessionId: string): ChatMessage[] {
  const stored = loadJson<ChatMessage[]>(messagesKey(sessionId), []);
  if (!Array.isArray(stored)) return [];
  return stored.filter(
    (item): item is ChatMessage =>
      typeof item === 'object' &&
      item !== null &&
      typeof item.id === 'string' &&
      (item.role === 'user' || item.role === 'assistant'),
  );
}

export function saveMessages(sessionId: string, messages: ChatMessage[]): void {
  saveJson(messagesKey(sessionId), messages.slice(-MAX_CACHED_MESSAGES));
}

export function forgetMessages(sessionId: string): void {
  removeKey(messagesKey(sessionId));
}

/** Rebuilds a plain transcript from the server session when no cache exists. */
export function messagesFromTurns(turns: Turn[]): ChatMessage[] {
  const base = Date.now() - turns.length;
  return turns.map((turn, index) =>
    turn.role === 'user'
      ? { id: randomId('turn'), role: 'user', content: turn.content, createdAt: base + index }
      : {
          id: randomId('turn'),
          role: 'assistant',
          content: turn.content,
          createdAt: base + index,
          result: null,
          error: null,
        },
  );
}

/** First question of a conversation, trimmed into a sidebar label. */
export function titleFromQuestion(question: string): string {
  const cleaned = question.replace(/\s+/g, ' ').trim();
  if (cleaned.length === 0) return NEW_SESSION_TITLE;
  return cleaned.length > 48 ? `${cleaned.slice(0, 47)}…` : cleaned;
}
