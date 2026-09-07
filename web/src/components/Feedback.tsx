import type { ReactElement, ReactNode } from 'react';

interface SpinnerProps {
  label?: string;
  /** Renders the label next to the spinner instead of only to screen readers. */
  inline?: boolean;
}

export function Spinner({ label = 'Loading', inline = false }: SpinnerProps): ReactElement {
  return (
    <span className="spinner-row" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span className={inline ? 'spinner-row__label' : 'sr-only'}>{label}</span>
    </span>
  );
}

interface EmptyStateProps {
  icon?: string;
  title: string;
  body?: ReactNode;
  action?: ReactNode;
}

export function EmptyState({ icon, title, body, action }: EmptyStateProps): ReactElement {
  return (
    <div className="empty">
      {icon !== undefined && (
        <div className="empty__icon" aria-hidden="true">
          {icon}
        </div>
      )}
      <p className="empty__title">{title}</p>
      {body !== undefined && <div className="empty__body">{body}</div>}
      {action !== undefined && <div className="empty__action">{action}</div>}
    </div>
  );
}

interface ErrorNoteProps {
  message: string;
  onDismiss?: () => void;
}

export function ErrorNote({ message, onDismiss }: ErrorNoteProps): ReactElement {
  return (
    <div className="error-note" role="alert">
      <span className="error-note__icon" aria-hidden="true">
        !
      </span>
      <span className="error-note__message">{message}</span>
      {onDismiss !== undefined && (
        <button type="button" className="icon-button" onClick={onDismiss} aria-label="Dismiss">
          ×
        </button>
      )}
    </div>
  );
}
