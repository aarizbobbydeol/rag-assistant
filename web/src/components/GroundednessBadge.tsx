import { useId, useState } from 'react';
import type { ReactElement } from 'react';

import type { Groundedness } from '../api/types';
import { formatPercent } from '../lib/answer';

/** Mirrors `Settings.min_answer_groundedness`: the strict-mode pass mark. */
const GOOD_ENOUGH = 0.6;
const SHAKY = 0.4;

type Verdict = 'abstained' | 'ungrounded' | 'partial' | 'grounded';

function verdictOf(groundedness: Groundedness): Verdict {
  if (groundedness.abstained) return 'abstained';
  if (groundedness.total_sentences === 0) return 'ungrounded';
  if (groundedness.score < SHAKY) return 'ungrounded';
  if (groundedness.score < GOOD_ENOUGH) return 'partial';
  return 'grounded';
}

const HEADLINE: Record<Verdict, string> = {
  abstained: 'Abstained',
  ungrounded: 'Not grounded',
  partial: 'Partly grounded',
  grounded: 'Grounded',
};

interface GroundednessBadgeProps {
  groundedness: Groundedness;
}

/**
 * The trust signal for an answer: how much of it the guard could tie back to
 * retrieved text. Anything below the strict-mode threshold is styled as a
 * warning rather than a neutral statistic, because that is the case a reader
 * must not skim past.
 */
export function GroundednessBadge({ groundedness }: GroundednessBadgeProps): ReactElement {
  const [open, setOpen] = useState(false);
  const verdict = verdictOf(groundedness);
  const warn = verdict === 'abstained' || verdict === 'ungrounded';
  // Several answers are on screen at once, so the panel id has to be unique.
  const detailId = useId();

  return (
    <div className={`ground ground--${verdict}`}>
      <button
        type="button"
        className="ground__summary"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-controls={detailId}
      >
        <span className="ground__dot" aria-hidden="true" />
        <span className="ground__headline">
          {warn ? '⚠ ' : ''}
          {HEADLINE[verdict]}
        </span>
        <span className="ground__score">{formatPercent(groundedness.score)}</span>
        <span className="ground__ratio">
          {groundedness.supported_sentences}/{groundedness.total_sentences} sentences supported
        </span>
        <span className="ground__chevron" aria-hidden="true">
          {open ? '▴' : '▾'}
        </span>
      </button>

      <div id={detailId} className="ground__detail" hidden={!open}>
        {groundedness.reason !== null && <p className="ground__reason">{groundedness.reason}</p>}
        <dl className="ground__stats">
          <div>
            <dt>Uncited sentences</dt>
            <dd>{groundedness.uncited_sentences}</dd>
          </div>
          <div>
            <dt>Invalid markers</dt>
            <dd>
              {groundedness.invalid_citations.length === 0
                ? 'none'
                : groundedness.invalid_citations.join(', ')}
            </dd>
          </div>
        </dl>
        {groundedness.unsupported.length > 0 && (
          <div className="ground__unsupported">
            <p className="ground__unsupported-title">Sentences with no support in the context</p>
            <ul>
              {groundedness.unsupported.map((sentence, index) => (
                <li key={`${index}-${sentence.slice(0, 24)}`}>{sentence}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
