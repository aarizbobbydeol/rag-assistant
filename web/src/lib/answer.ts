/**
 * Splits an answer into prose and citation markers.
 *
 * `app.generation.citations` emits `[1]`, `[1,2]` and `[1][2]`; the backend has
 * already renumbered survivors into a dense 1..n sequence, so a marker that does
 * not resolve here is genuinely unbacked and is rendered as such rather than
 * hidden.
 */

import type { Citation, ScoredChunk } from '../api/types';

const MARKER_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

export interface TextSegment {
  kind: 'text';
  text: string;
}

export interface MarkerSegment {
  kind: 'markers';
  markers: number[];
}

export type AnswerSegment = TextSegment | MarkerSegment;

export function parseAnswer(answer: string): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  let cursor = 0;

  // `exec` in a loop rather than `matchAll` so the gap between matches is easy
  // to slice out verbatim, whitespace included.
  MARKER_RE.lastIndex = 0;
  let match = MARKER_RE.exec(answer);
  while (match !== null) {
    if (match.index > cursor) {
      segments.push({ kind: 'text', text: answer.slice(cursor, match.index) });
    }
    const markers = (match[1] ?? '')
      .split(',')
      .map((part) => Number.parseInt(part.trim(), 10))
      .filter((value) => Number.isInteger(value));
    segments.push({ kind: 'markers', markers });
    cursor = match.index + match[0].length;
    match = MARKER_RE.exec(answer);
  }

  if (cursor < answer.length) {
    segments.push({ kind: 'text', text: answer.slice(cursor) });
  }
  return segments;
}

export function findCitation(citations: Citation[], marker: number): Citation | null {
  return citations.find((citation) => citation.marker === marker) ?? null;
}

export function findContext(contexts: ScoredChunk[], chunkId: string): ScoredChunk | null {
  return contexts.find((context) => context.chunk.chunk_id === chunkId) ?? null;
}

/**
 * Locates the cited quote inside the full passage so the source panel can show
 * the surrounding text with the quote highlighted. Falls back to no highlight
 * when the quote was normalised differently from the stored chunk.
 */
export function splitAroundQuote(
  passage: string,
  quote: string,
): { before: string; hit: string; after: string } | null {
  const needle = quote.trim();
  if (needle.length < 8) return null;
  const index = passage.toLowerCase().indexOf(needle.toLowerCase());
  if (index < 0) return null;
  return {
    before: passage.slice(0, index),
    hit: passage.slice(index, index + needle.length),
    after: passage.slice(index + needle.length),
  };
}

/** `0.8123` -> `0.812`; scores are cosine-ish and read better at 3 decimals. */
export function formatScore(score: number): string {
  return Number.isFinite(score) ? score.toFixed(3) : '—';
}

export function formatPercent(value: number): string {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—';
}
