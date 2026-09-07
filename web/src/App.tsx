import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactElement } from 'react';

import {
  ApiError,
  chat,
  describeError,
  getSession,
  health,
  isAbort,
  logout,
  resetSession,
} from './api/client';
import type { HealthResponse, IngestResponse } from './api/types';
import { ChatPanel } from './components/ChatPanel';
import type { CitationSelection } from './components/ChatPanel';
import { ErrorNote } from './components/Feedback';
import { LoginPanel } from './components/LoginPanel';
import { SessionList } from './components/SessionList';
import { SettingsDrawer } from './components/SettingsDrawer';
import { SourcePanel } from './components/SourcePanel';
import { UploadPanel } from './components/UploadPanel';
import { useAuthToken } from './hooks/useAuthToken';
import type { ChatMessage, SessionMeta } from './lib/conversations';
import {
  forgetMessages,
  loadMessages,
  loadSessions,
  messagesFromTurns,
  newSession,
  saveMessages,
  saveSessions,
  titleFromQuestion,
  NEW_SESSION_TITLE,
} from './lib/conversations';
import type { ChatSettings } from './lib/settings';
import { loadSettings, saveSettings } from './lib/settings';
import { randomId } from './lib/storage';

/** `undefined` for a session means "transcript not loaded yet". */
type TranscriptMap = Record<string, ChatMessage[] | undefined>;

type AuthPrompt = 'hidden' | 'required' | 'optional';

export function App(): ReactElement {
  const token = useAuthToken();

  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [transcripts, setTranscripts] = useState<TranscriptMap>({});

  const [settings, setSettings] = useState<ChatSettings>(loadSettings);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [service, setService] = useState<HealthResponse | null>(null);
  const [authPrompt, setAuthPrompt] = useState<AuthPrompt>('hidden');
  const [banner, setBanner] = useState<string | null>(null);

  const [pending, setPending] = useState(false);
  const [selection, setSelection] = useState<CitationSelection | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const askAbort = useRef<AbortController | null>(null);
  const hydrated = useRef<Set<string>>(new Set());

  const messages = useMemo<ChatMessage[]>(
    () => (activeId === null ? [] : (transcripts[activeId] ?? [])),
    [activeId, transcripts],
  );

  // -- boot ---------------------------------------------------------------- //
  useEffect(() => {
    const stored = loadSessions();
    setSessions(stored.length > 0 ? stored : [newSession()]);
  }, []);

  // Keeps the selected conversation valid after a boot, a delete, or storage
  // that came back empty.
  useEffect(() => {
    if (sessions.length === 0) return;
    if (activeId !== null && sessions.some((session) => session.id === activeId)) return;
    setActiveId(sessions[0].id);
    setSelection(null);
  }, [sessions, activeId]);

  const refreshHealth = useCallback(async (signal?: AbortSignal): Promise<void> => {
    try {
      setService(await health(signal));
      setBanner(null);
    } catch (error) {
      if (isAbort(error)) return;
      if (error instanceof ApiError && error.status === 401) {
        setAuthPrompt('required');
        return;
      }
      setService(null);
      setBanner(describeError(error));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refreshHealth(controller.signal);
    return () => controller.abort();
  }, [refreshHealth, token]);

  // -- transcript hydration ------------------------------------------------ //
  useEffect(() => {
    const sessionId = activeId;
    if (sessionId === null || hydrated.current.has(sessionId)) return;
    hydrated.current.add(sessionId);

    const cached = loadMessages(sessionId);
    if (cached.length > 0) {
      setTranscripts((current) => ({ ...current, [sessionId]: cached }));
      return;
    }

    // Cold cache (another browser, cleared storage): the server still holds the
    // transcript for the session id, just without citations.
    void getSession(sessionId)
      .then((response) => {
        setTranscripts((current) => ({
          ...current,
          [sessionId]: messagesFromTurns(response.turns),
        }));
      })
      .catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 401) {
          setAuthPrompt('required');
        }
        setTranscripts((current) => ({ ...current, [sessionId]: [] }));
      });
  }, [activeId]);

  // -- persistence --------------------------------------------------------- //
  useEffect(() => {
    if (activeId === null) return;
    const current = transcripts[activeId];
    if (current === undefined) return;
    saveMessages(activeId, current);
  }, [activeId, transcripts]);

  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  useEffect(() => {
    if (sessions.length === 0) return;
    saveSessions(sessions);
  }, [sessions]);

  // -- session actions ----------------------------------------------------- //
  const touchSession = useCallback((sessionId: string, title?: string): void => {
    const now = Date.now();
    setSessions((current) =>
      current
        .map((session) =>
          session.id === sessionId
            ? {
                ...session,
                updatedAt: now,
                title:
                  title !== undefined && session.title === NEW_SESSION_TITLE
                    ? title
                    : session.title,
              }
            : session,
        )
        .sort((a, b) => b.updatedAt - a.updatedAt),
    );
  }, []);

  const createSession = useCallback((): void => {
    const session = newSession();
    hydrated.current.add(session.id);
    setTranscripts((current) => ({ ...current, [session.id]: [] }));
    setSessions((current) => [session, ...current]);
    setActiveId(session.id);
    setSelection(null);
    setSidebarOpen(false);
  }, []);

  const selectSession = useCallback((sessionId: string): void => {
    setActiveId(sessionId);
    setSelection(null);
    setSidebarOpen(false);
  }, []);

  const deleteSession = useCallback((sessionId: string): void => {
    forgetMessages(sessionId);
    hydrated.current.delete(sessionId);
    // Best effort: the server-side transcript is LRU/TTL anyway, so a failure
    // here costs nothing the user can see.
    void resetSession(sessionId).catch(() => undefined);
    setTranscripts((current) => {
      const next = { ...current };
      delete next[sessionId];
      return next;
    });
    // Deleting the last conversation leaves an empty one behind rather than an
    // empty sidebar; `fresh` is built outside the updater to keep it pure.
    const fresh = newSession();
    setSessions((current) => {
      const remaining = current.filter((session) => session.id !== sessionId);
      return remaining.length > 0 ? remaining : [fresh];
    });
  }, []);

  // -- asking -------------------------------------------------------------- //
  const appendMessage = useCallback((sessionId: string, message: ChatMessage): void => {
    setTranscripts((current) => ({
      ...current,
      [sessionId]: [...(current[sessionId] ?? []), message],
    }));
  }, []);

  const ask = useCallback(
    async (question: string): Promise<void> => {
      const sessionId = activeId;
      if (sessionId === null || pending) return;

      appendMessage(sessionId, {
        id: randomId('msg'),
        role: 'user',
        content: question,
        createdAt: Date.now(),
      });
      touchSession(sessionId, titleFromQuestion(question));
      setSelection(null);
      setPending(true);

      const controller = new AbortController();
      askAbort.current = controller;
      try {
        const result = await chat(
          {
            question,
            session_id: sessionId,
            top_k: settings.topK,
            mode: settings.mode,
            rerank: settings.rerank,
            strict: settings.strict,
            include_contexts: true,
          },
          controller.signal,
        );
        appendMessage(sessionId, {
          id: randomId('msg'),
          role: 'assistant',
          content: result.answer,
          createdAt: Date.now(),
          result,
          error: null,
        });
      } catch (error) {
        if (isAbort(error)) {
          appendMessage(sessionId, {
            id: randomId('msg'),
            role: 'assistant',
            content: '',
            createdAt: Date.now(),
            result: null,
            error: 'Cancelled before an answer came back.',
          });
        } else {
          if (error instanceof ApiError && error.status === 401) {
            setAuthPrompt('required');
          }
          appendMessage(sessionId, {
            id: randomId('msg'),
            role: 'assistant',
            content: '',
            createdAt: Date.now(),
            result: null,
            error: describeError(error),
          });
        }
      } finally {
        askAbort.current = null;
        setPending(false);
      }
    },
    [activeId, appendMessage, pending, settings, touchSession],
  );

  const cancelAsk = useCallback((): void => {
    askAbort.current?.abort();
  }, []);

  const handleIngested = useCallback(
    (response: IngestResponse): void => {
      setService((current) =>
        current === null ? current : { ...current, index_size: response.index_size },
      );
      void refreshHealth();
    },
    [refreshHealth],
  );

  const handleSignOut = useCallback((): void => {
    logout();
    setAuthPrompt('hidden');
  }, []);

  // -- derived ------------------------------------------------------------- //
  const selected = useMemo(() => {
    if (selection === null) return null;
    const message = messages.find((item) => item.id === selection.messageId);
    if (message === undefined || message.role !== 'assistant' || message.result === null) {
      return null;
    }
    return message.result;
  }, [messages, selection]);

  if (authPrompt !== 'hidden') {
    return (
      <LoginPanel
        optional={authPrompt === 'optional'}
        onAuthenticated={() => {
          setAuthPrompt('hidden');
          void refreshHealth();
        }}
        onSkip={() => setAuthPrompt('hidden')}
      />
    );
  }

  return (
    <div className={`shell${selected === null ? '' : ' shell--with-sources'}`}>
      <header className="topbar">
        <button
          type="button"
          className="icon-button topbar__menu"
          onClick={() => setSidebarOpen((open) => !open)}
          aria-label="Toggle sidebar"
          aria-expanded={sidebarOpen}
        >
          ☰
        </button>
        <div className="topbar__brand">
          <span className="topbar__mark" aria-hidden="true">
            ¶
          </span>
          <span className="topbar__name">RAG Assistant</span>
        </div>

        <div className="topbar__stats">
          {service === null ? (
            <span className="pill pill--warn">API offline</span>
          ) : (
            <>
              <span className="pill pill--quiet">{service.index_size} chunks</span>
              <span className="pill pill--quiet">{service.documents} docs</span>
              <span className={`pill ${service.llm_available ? 'pill--ok' : 'pill--warn'}`}>
                {service.llm_provider}
              </span>
            </>
          )}
        </div>

        <div className="topbar__actions">
          <button
            type="button"
            className="button button--small"
            onClick={() => setSettingsOpen(true)}
          >
            Settings
          </button>
          {token === null ? (
            <button
              type="button"
              className="button button--small"
              onClick={() => setAuthPrompt('optional')}
            >
              Sign in
            </button>
          ) : (
            <button type="button" className="button button--small" onClick={handleSignOut}>
              Sign out
            </button>
          )}
        </div>
      </header>

      {banner !== null && (
        <div className="shell__banner">
          <ErrorNote message={banner} onDismiss={() => setBanner(null)} />
        </div>
      )}

      <div className="shell__body">
        <div className={`sidebar${sidebarOpen ? ' sidebar--open' : ''}`}>
          <SessionList
            sessions={sessions}
            activeId={activeId}
            onSelect={selectSession}
            onCreate={createSession}
            onDelete={deleteSession}
          />
          <UploadPanel onIngested={handleIngested} />
        </div>

        <main className="main">
          <ChatPanel
            messages={messages}
            pending={pending}
            hasDocuments={service !== null && service.index_size > 0}
            settings={settings}
            selection={selection}
            onAsk={(question) => void ask(question)}
            onCancel={cancelAsk}
            onSelectCitation={setSelection}
          />
        </main>

        {selected !== null && (
          <SourcePanel
            citations={selected.citations}
            contexts={selected.contexts}
            activeMarker={selection === null ? null : selection.marker}
            onSelectMarker={(marker) =>
              setSelection((current) =>
                current === null ? current : { ...current, marker },
              )
            }
            onClose={() => setSelection(null)}
          />
        )}
      </div>

      <SettingsDrawer
        open={settingsOpen}
        settings={settings}
        health={service}
        onChange={setSettings}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
