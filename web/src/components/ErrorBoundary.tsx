import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/**
 * Last line of defence around the whole tree.
 *
 * A render crash in one panel would otherwise blank the page with no way back;
 * this keeps the conversation cache intact and offers a retry that re-mounts the
 * children without a full reload.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('Unhandled UI error', error, info.componentStack);
  }

  private readonly handleRetry = (): void => {
    this.setState({ error: null });
  };

  private readonly handleReload = (): void => {
    window.location.reload();
  };

  override render(): ReactNode {
    const { error } = this.state;
    if (error === null) {
      return this.props.children;
    }
    return (
      <div className="crash" role="alert">
        <div className="crash__card">
          <h1 className="crash__title">The interface hit an unexpected error</h1>
          <p className="crash__body">
            Your conversations are stored locally and were not lost. Retrying re-renders the app; a
            reload starts it from scratch.
          </p>
          <pre className="crash__detail">{error.message}</pre>
          <div className="crash__actions">
            <button type="button" className="button button--primary" onClick={this.handleRetry}>
              Retry
            </button>
            <button type="button" className="button" onClick={this.handleReload}>
              Reload the page
            </button>
          </div>
        </div>
      </div>
    );
  }
}
