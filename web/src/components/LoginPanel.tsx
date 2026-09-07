import { useState } from 'react';
import type { FormEvent, ReactElement } from 'react';

import { describeError, login, register } from '../api/client';
import { ErrorNote, Spinner } from './Feedback';

/** Mirrors `Settings.auth_min_password_length`. */
const MIN_PASSWORD_LENGTH = 8;

type Mode = 'login' | 'register';

interface LoginPanelProps {
  /** Called once a token is in hand; the parent re-probes the service state. */
  onAuthenticated: () => void;
  /**
   * True when the API answered without a token, i.e. `RAG_AUTH_REQUIRED=false`.
   * Signing in is then optional and the panel offers a way past it.
   */
  optional: boolean;
  onSkip: () => void;
}

export function LoginPanel({ onAuthenticated, optional, onSkip }: LoginPanelProps): ReactElement {
  const [mode, setMode] = useState<Mode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const canSubmit =
    email.trim().length > 0 && password.length >= MIN_PASSWORD_LENGTH && !busy;

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      if (mode === 'register') {
        await register(email.trim(), password);
        setNotice('Account created. Signing you in…');
      }
      // Registration does not return a token, so both paths finish here.
      await login(email.trim(), password);
      onAuthenticated();
    } catch (cause) {
      setError(describeError(cause));
      setNotice(null);
    } finally {
      setBusy(false);
    }
  }

  function switchMode(next: Mode): void {
    setMode(next);
    setError(null);
    setNotice(null);
  }

  return (
    <div className="auth">
      <form
        className="auth__card"
        onSubmit={(event) => {
          void handleSubmit(event);
        }}
      >
        <div className="auth__brand">
          <span className="auth__mark" aria-hidden="true">
            ¶
          </span>
          <div>
            <h1 className="auth__title">RAG Assistant</h1>
            <p className="auth__subtitle">Answers from your documents, with citations you can check.</p>
          </div>
        </div>

        <div className="auth__tabs" role="tablist" aria-label="Authentication mode">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'login'}
            className={`auth__tab${mode === 'login' ? ' auth__tab--active' : ''}`}
            onClick={() => switchMode('login')}
          >
            Sign in
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'register'}
            className={`auth__tab${mode === 'register' ? ' auth__tab--active' : ''}`}
            onClick={() => switchMode('register')}
          >
            Create account
          </button>
        </div>

        <label className="field">
          <span className="field__label">Email</span>
          <input
            className="field__input"
            type="email"
            name="email"
            autoComplete="username"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@example.com"
          />
        </label>

        <label className="field">
          <span className="field__label">Password</span>
          <input
            className="field__input"
            type="password"
            name="password"
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            required
            minLength={MIN_PASSWORD_LENGTH}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="At least 8 characters"
          />
          <span className="field__hint">
            {mode === 'register'
              ? `Minimum ${MIN_PASSWORD_LENGTH} characters. Stored as a PBKDF2-SHA256 hash.`
              : 'Tokens are HS256 JWTs and expire on the server schedule.'}
          </span>
        </label>

        {error !== null && <ErrorNote message={error} onDismiss={() => setError(null)} />}
        {notice !== null && <p className="auth__notice">{notice}</p>}

        <button type="submit" className="button button--primary button--block" disabled={!canSubmit}>
          {busy ? <Spinner label="Working" inline /> : mode === 'login' ? 'Sign in' : 'Create account'}
        </button>

        {optional && (
          <button type="button" className="button button--ghost button--block" onClick={onSkip}>
            Continue without signing in
          </button>
        )}
      </form>
    </div>
  );
}
