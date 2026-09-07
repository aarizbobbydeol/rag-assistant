import { useState } from 'react';
import type { ReactElement } from 'react';

import type { Citation, ScoredChunk } from '../api/types';
import { findCitation, findContext, formatScore, splitAroundQuote } from '../lib/answer';
import { EmptyState } from './Feedback';

interface ScoreBarProps {
  label: string;
  value: number;
}

function ScoreBar({ label, value }: ScoreBarProps): ReactElement {
  // Cosine and fused scores land in [-1, 1]; map that onto the bar width so a
  // negative score reads as "barely relevant" instead of overflowing.
  const width = Math.max(0, Math.min(1, (value + 1) / 2)) * 100;
  return (
    <div className="scorebar">
      <span className="scorebar__label">{label}</span>
      <span className="scorebar__track">
        <span className="scorebar__fill" style={{ width: `${width}%` }} />
      </span>
      <span className="scorebar__value">{formatScore(value)}</span>
    </div>
  );
}

interface PassageProps {
  passage: string;
  quote: string;
}

function Passage({ passage, quote }: PassageProps): ReactElement {
  const split = splitAroundQuote(passage, quote);
  if (split === null) {
    return <p className="passage">{passage}</p>;
  }
  return (
    <p className="passage">
      {split.before}
      <mark className="passage__hit">{split.hit}</mark>
      {split.after}
    </p>
  );
}

interface SourceDetailProps {
  citation: Citation;
  context: ScoredChunk | null;
}

function SourceDetail({ citation, context }: SourceDetailProps): ReactElement {
  const chunk: ScoredChunk['chunk'] | null = context === null ? null : context.chunk;
  const passage = chunk === null ? citation.quote : chunk.text;
  const dense: number | null = context === null ? null : context.dense_score;
  const lexical: number | null = context === null ? null : context.lexical_score;
  const rerank: number | null = context === null ? null : context.rerank_score;

  return (
    <article className="source">
      <header className="source__header">
        <span className="source__marker" aria-hidden="true">
          {citation.marker}
        </span>
        <div className="source__ident">
          <h3 className="source__title">{citation.title}</h3>
          <p className="source__path" title={citation.source}>
            {citation.source}
          </p>
        </div>
      </header>

      <div className="source__facts">
        <span className="pill">
          {citation.page === null
            ? chunk === null
              ? 'passage'
              : `chunk ${chunk.ordinal}`
            : `page ${citation.page}`}
        </span>
        {context !== null && <span className="pill pill--quiet">rank {context.rank}</span>}
        {context !== null && <span className="pill pill--quiet">{context.retriever}</span>}
        {chunk !== null && <span className="pill pill--quiet">{chunk.token_count} tokens</span>}
      </div>

      <div className="source__scores">
        <ScoreBar label="final" value={context === null ? citation.score : context.score} />
        {dense !== null && <ScoreBar label="dense" value={dense} />}
        {lexical !== null && <ScoreBar label="bm25" value={lexical} />}
        {rerank !== null && <ScoreBar label="rerank" value={rerank} />}
      </div>

      {citation.quote.trim().length > 0 && (
        <blockquote className="source__quote">{citation.quote}</blockquote>
      )}

      <details className="source__full" open={passage !== citation.quote}>
        <summary>Full passage</summary>
        <Passage passage={passage} quote={citation.quote} />
      </details>
    </article>
  );
}

interface SourcePanelProps {
  citations: Citation[];
  contexts: ScoredChunk[];
  activeMarker: number | null;
  onSelectMarker: (marker: number) => void;
  onClose: () => void;
}

/**
 * The side panel behind every `[n]` pill: identity, retrieval scores and the
 * exact text the answer leaned on, so a claim can be checked without leaving
 * the conversation.
 */
export function SourcePanel({
  citations,
  contexts,
  activeMarker,
  onSelectMarker,
  onClose,
}: SourcePanelProps): ReactElement {
  const [showAll, setShowAll] = useState(false);
  const first: Citation | undefined = citations[0];
  const active: Citation | null =
    activeMarker === null
      ? first === undefined
        ? null
        : first
      : findCitation(citations, activeMarker);

  const citedChunkIds = new Set(citations.map((citation) => citation.chunk_id));
  const uncited = contexts.filter((context) => !citedChunkIds.has(context.chunk.chunk_id));

  return (
    <aside className="sources" aria-label="Cited sources">
      <div className="sources__header">
        <h2 className="sources__title">Sources</h2>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close sources">
          ×
        </button>
      </div>

      {citations.length === 0 ? (
        <EmptyState
          icon="◌"
          title="No citations on this answer"
          body="The model answered without pointing at a retrieved passage — treat it with suspicion."
        />
      ) : (
        <>
          <div className="sources__tabs" role="tablist" aria-label="Citations">
            {citations.map((citation) => (
              <button
                key={citation.marker}
                type="button"
                role="tab"
                aria-selected={active?.marker === citation.marker}
                className={`sources__tab${
                  active?.marker === citation.marker ? ' sources__tab--active' : ''
                }`}
                onClick={() => onSelectMarker(citation.marker)}
                title={citation.title}
              >
                {citation.marker}
              </button>
            ))}
          </div>

          {active !== null && (
            <SourceDetail citation={active} context={findContext(contexts, active.chunk_id)} />
          )}
        </>
      )}

      {uncited.length > 0 && (
        <div className="sources__extra">
          <button
            type="button"
            className="button button--ghost button--small"
            onClick={() => setShowAll((current) => !current)}
            aria-expanded={showAll}
          >
            {showAll ? 'Hide' : 'Show'} {uncited.length} retrieved but uncited passage
            {uncited.length === 1 ? '' : 's'}
          </button>
          {showAll && (
            <ul className="sources__extra-list">
              {uncited.map((context) => (
                <li key={context.chunk.chunk_id} className="sources__extra-item">
                  <p className="sources__extra-title">
                    {context.chunk.title}
                    <span className="sources__extra-score">{formatScore(context.score)}</span>
                  </p>
                  <p className="sources__extra-text">{context.chunk.text}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </aside>
  );
}
