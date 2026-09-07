import { useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent, ReactElement } from 'react';

import type { AnswerResult } from '../api/types';
import type { ChatMessage } from '../lib/conversations';
import type { ChatSettings } from '../lib/settings';
import { AnswerBody } from './AnswerBody';
import { EmptyState, ErrorNote, Spinner } from './Feedback';
import { GroundednessBadge } from './GroundednessBadge';

const EXAMPLE_QUESTIONS = [
  'What does the refund policy say about digital downloads?',
  'Summarise the exceptions and cite them.',
  'How long do approved refunds take to settle?',
];

export interface CitationSelection {
  messageId: string;
  marker: number;
}

interface ChatPanelProps {
  messages: ChatMessage[];
  pending: boolean;
  hasDocuments: boolean;
  settings: ChatSettings;
  selection: CitationSelection | null;
  onAsk: (question: string) => void;
  onCancel: () => void;
  onSelectCitation: (selection: CitationSelection) => void;
}

function latencyLabel(result: AnswerResult): string | null {
  const total = result.latency_ms.total;
  return typeof total === 'number' ? `${Math.round(total)} ms` : null;
}

export function ChatPanel({
  messages,
  pending,
  hasDocuments,
  settings,
  selection,
  onAsk,
  onCancel,
  onSelectCitation,
}: ChatPanelProps): ReactElement {
  const [draft, setDraft] = useState('');
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages.length, pending]);

  function submit(): void {
    const question = draft.trim();
    if (question.length === 0 || pending) return;
    setDraft('');
    onAsk(question);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    submit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    // Enter sends, Shift+Enter is a newline — the convention people expect from
    // a chat composer.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <section className="chat" aria-label="Conversation">
      <div className="chat__scroll">
        {messages.length === 0 && !pending && (
          <EmptyState
            icon="✦"
            title={hasDocuments ? 'Ask something about your documents' : 'Index a document first'}
            body={
              hasDocuments ? (
                <ul className="chat__examples">
                  {EXAMPLE_QUESTIONS.map((example) => (
                    <li key={example}>
                      <button
                        type="button"
                        className="chat__example"
                        onClick={() => onAsk(example)}
                      >
                        {example}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                'Drop a PDF, Markdown or text file into the Documents panel and it will be chunked, embedded and indexed.'
              )
            }
          />
        )}

        {messages.map((message) =>
          message.role === 'user' ? (
            <article key={message.id} className="bubble bubble--user">
              <p className="bubble__text">{message.content}</p>
            </article>
          ) : (
            <article key={message.id} className="bubble bubble--assistant">
              {message.error !== null ? (
                <ErrorNote message={message.error} />
              ) : (
                <>
                  <AnswerBody
                    text={message.content}
                    citations={message.result?.citations ?? []}
                    activeMarker={
                      selection !== null && selection.messageId === message.id
                        ? selection.marker
                        : null
                    }
                    onSelectMarker={(marker) =>
                      onSelectCitation({ messageId: message.id, marker })
                    }
                  />
                  {message.result !== null && (
                    <footer className="bubble__footer">
                      <GroundednessBadge groundedness={message.result.groundedness} />
                      <div className="bubble__meta">
                        <span>{message.result.citations.length} citations</span>
                        <span>{message.result.contexts.length} contexts</span>
                        {latencyLabel(message.result) !== null && (
                          <span>{latencyLabel(message.result)}</span>
                        )}
                        {message.result.usage.total_tokens > 0 && (
                          <span>{message.result.usage.total_tokens} tokens</span>
                        )}
                      </div>
                    </footer>
                  )}
                </>
              )}
            </article>
          ),
        )}

        {pending && (
          <article className="bubble bubble--assistant bubble--pending">
            <Spinner label="Retrieving and composing an answer" inline />
          </article>
        )}

        <div ref={bottomRef} />
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          className="composer__input"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the indexed documents…"
          rows={2}
          aria-label="Your question"
        />
        <div className="composer__actions">
          <span className="composer__hint">
            {settings.mode} · top_k {settings.topK}
            {settings.rerank ? ' · rerank' : ''}
            {settings.strict ? ' · strict' : ''}
          </span>
          {pending ? (
            <button type="button" className="button" onClick={onCancel}>
              Stop
            </button>
          ) : (
            <button
              type="submit"
              className="button button--primary"
              disabled={draft.trim().length === 0}
            >
              Ask
            </button>
          )}
        </div>
      </form>
    </section>
  );
}
