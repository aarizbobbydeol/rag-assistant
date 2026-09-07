import type { ReactElement } from 'react';

import type { SessionMeta } from '../lib/conversations';

interface SessionListProps {
  sessions: SessionMeta[];
  activeId: string | null;
  onSelect: (sessionId: string) => void;
  onCreate: () => void;
  onDelete: (sessionId: string) => void;
}

function relativeTime(timestamp: number): string {
  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 60) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function SessionList({
  sessions,
  activeId,
  onSelect,
  onCreate,
  onDelete,
}: SessionListProps): ReactElement {
  return (
    <section className="panel sessions" aria-labelledby="sessions-heading">
      <div className="panel__header">
        <h2 id="sessions-heading" className="panel__title">
          Conversations
        </h2>
        <button type="button" className="button button--small" onClick={onCreate}>
          + New
        </button>
      </div>

      {sessions.length === 0 ? (
        <p className="sessions__empty">No conversations yet.</p>
      ) : (
        <ul className="sessions__list">
          {sessions.map((session) => (
            <li key={session.id}>
              <div
                className={`sessions__item${
                  session.id === activeId ? ' sessions__item--active' : ''
                }`}
              >
                <button
                  type="button"
                  className="sessions__open"
                  onClick={() => onSelect(session.id)}
                  aria-current={session.id === activeId}
                >
                  <span className="sessions__label">{session.title}</span>
                  <span className="sessions__time">{relativeTime(session.updatedAt)}</span>
                </button>
                <button
                  type="button"
                  className="icon-button sessions__delete"
                  onClick={() => onDelete(session.id)}
                  aria-label={`Delete conversation: ${session.title}`}
                  title="Delete conversation"
                >
                  ×
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
