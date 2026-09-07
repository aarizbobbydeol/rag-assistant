import { Fragment } from 'react';
import type { ReactElement } from 'react';

import type { Citation } from '../api/types';
import { findCitation, parseAnswer } from '../lib/answer';

interface CitationPillProps {
  marker: number;
  citation: Citation | null;
  active: boolean;
  onSelect: (marker: number) => void;
}

function CitationPill({ marker, citation, active, onSelect }: CitationPillProps): ReactElement {
  // A marker the backend could not resolve is shown, not swallowed: a visible
  // dead citation is honest, a hidden one looks like an uncited claim.
  const resolved = citation !== null;
  const label =
    citation === null
      ? `Source ${marker} could not be resolved`
      : `Source ${marker}: ${citation.title}${citation.page === null ? '' : `, page ${citation.page}`}`;

  return (
    <button
      type="button"
      className={[
        'cite',
        active ? 'cite--active' : '',
        resolved ? '' : 'cite--broken',
      ]
        .filter(Boolean)
        .join(' ')}
      onClick={() => onSelect(marker)}
      disabled={!resolved}
      title={label}
      aria-label={label}
      aria-pressed={active}
    >
      {marker}
    </button>
  );
}

interface AnswerBodyProps {
  text: string;
  citations: Citation[];
  activeMarker: number | null;
  onSelectMarker: (marker: number) => void;
}

/**
 * Renders the answer with its `[n]` markers replaced by clickable superscript
 * pills. Prose is emitted verbatim inside a `pre-wrap` container so the model's
 * paragraph breaks survive without a markdown dependency.
 */
export function AnswerBody({
  text,
  citations,
  activeMarker,
  onSelectMarker,
}: AnswerBodyProps): ReactElement {
  const segments = parseAnswer(text);

  return (
    <div className="answer">
      {segments.map((segment, index) =>
        segment.kind === 'text' ? (
          <Fragment key={`t${index}`}>{segment.text}</Fragment>
        ) : (
          <sup className="cite-group" key={`m${index}`}>
            {segment.markers.map((marker) => (
              <CitationPill
                key={marker}
                marker={marker}
                citation={findCitation(citations, marker)}
                active={activeMarker === marker}
                onSelect={onSelectMarker}
              />
            ))}
          </sup>
        ),
      )}
    </div>
  );
}
