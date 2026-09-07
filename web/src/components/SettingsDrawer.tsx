import { useEffect, useRef } from 'react';
import type { ReactElement } from 'react';

import type { HealthResponse } from '../api/types';
import type { ChatSettings } from '../lib/settings';
import {
  DEFAULT_SETTINGS,
  RETRIEVAL_MODES,
  RETRIEVAL_MODE_HELP,
  TOP_K_MAX,
  TOP_K_MIN,
} from '../lib/settings';

interface SettingsDrawerProps {
  open: boolean;
  settings: ChatSettings;
  health: HealthResponse | null;
  onChange: (settings: ChatSettings) => void;
  onClose: () => void;
}

export function SettingsDrawer({
  open,
  settings,
  health,
  onChange,
  onClose,
}: SettingsDrawerProps): ReactElement | null {
  const closeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return undefined;
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  function patch(changes: Partial<ChatSettings>): void {
    onChange({ ...settings, ...changes });
  }

  return (
    <div className="drawer-backdrop" onClick={onClose} role="presentation">
      <div
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-heading"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="drawer__header">
          <h2 id="settings-heading" className="drawer__title">
            Retrieval settings
          </h2>
          <button
            ref={closeRef}
            type="button"
            className="icon-button"
            onClick={onClose}
            aria-label="Close settings"
          >
            ×
          </button>
        </div>

        <div className="drawer__body">
          <div className="setting">
            <div className="setting__row">
              <label className="setting__label" htmlFor="top-k">
                Contexts (top_k)
              </label>
              <output className="setting__value" htmlFor="top-k">
                {settings.topK}
              </output>
            </div>
            <input
              id="top-k"
              type="range"
              min={TOP_K_MIN}
              max={TOP_K_MAX}
              step={1}
              value={settings.topK}
              onChange={(event) => patch({ topK: Number(event.target.value) })}
            />
            <p className="setting__hint">
              How many passages are handed to the model. More context costs tokens and can dilute
              precision.
            </p>
          </div>

          <fieldset className="setting">
            <legend className="setting__label">Retrieval mode</legend>
            <div className="segmented">
              {RETRIEVAL_MODES.map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={`segmented__option${
                    settings.mode === mode ? ' segmented__option--active' : ''
                  }`}
                  onClick={() => patch({ mode })}
                  aria-pressed={settings.mode === mode}
                >
                  {mode}
                </button>
              ))}
            </div>
            <p className="setting__hint">{RETRIEVAL_MODE_HELP[settings.mode]}</p>
          </fieldset>

          <div className="setting">
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.rerank}
                onChange={(event) => patch({ rerank: event.target.checked })}
              />
              <span className="toggle__track" aria-hidden="true">
                <span className="toggle__thumb" />
              </span>
              <span className="toggle__label">Rerank candidates</span>
            </label>
            <p className="setting__hint">
              Reorders the fused candidate list before the prompt is built
              {health === null ? '' : ` (${health.reranker})`}.
            </p>
          </div>

          <div className="setting">
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.strict}
                onChange={(event) => patch({ strict: event.target.checked })}
              />
              <span className="toggle__track" aria-hidden="true">
                <span className="toggle__thumb" />
              </span>
              <span className="toggle__label">Strict grounding</span>
            </label>
            <p className="setting__hint">
              Abstain instead of returning an answer the guard cannot tie back to the retrieved
              text.
            </p>
          </div>

          <button
            type="button"
            className="button button--ghost button--block"
            onClick={() => onChange(DEFAULT_SETTINGS)}
          >
            Reset to server defaults
          </button>

          {health !== null && (
            <dl className="drawer__facts">
              <div>
                <dt>Embedder</dt>
                <dd>{health.embedder}</dd>
              </div>
              <div>
                <dt>Vector store</dt>
                <dd>{health.vector_store}</dd>
              </div>
              <div>
                <dt>Generator</dt>
                <dd>
                  {health.llm_provider} / {health.llm_model}
                  {health.llm_available ? '' : ' (offline)'}
                </dd>
              </div>
              <div>
                <dt>Index</dt>
                <dd>
                  {health.index_size} chunks · {health.documents} documents
                </dd>
              </div>
            </dl>
          )}
        </div>
      </div>
    </div>
  );
}
