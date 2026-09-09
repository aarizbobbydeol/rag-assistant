import { useCallback, useRef, useState } from 'react';
import type { ChangeEvent, DragEvent, ReactElement } from 'react';

import { describeError, isAbort, uploadFile } from '../api/client';
import type { IngestResponse } from '../api/types';
import { SUPPORTED_SUFFIXES } from '../api/types';
import { randomId } from '../lib/storage';

/** Mirrors `Settings.max_upload_mb`; rejected client-side to save the round trip. */
const MAX_UPLOAD_MB = 32;

type ItemStatus = 'queued' | 'uploading' | 'done' | 'error';

interface UploadItem {
  id: string;
  name: string;
  size: number;
  status: ItemStatus;
  percent: number;
  chunks: number;
  error: string | null;
}

interface UploadPanelProps {
  /** Fired after every successful file so the header can refresh index stats. */
  onIngested: (response: IngestResponse) => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function suffixOf(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot < 0 ? '' : name.slice(dot).toLowerCase();
}

function rejectionReason(file: File): string | null {
  if (!(SUPPORTED_SUFFIXES as readonly string[]).includes(suffixOf(file.name))) {
    return `Unsupported file type (${suffixOf(file.name) || 'no extension'}).`;
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `Larger than the ${MAX_UPLOAD_MB} MB limit.`;
  }
  return null;
}

/** Turn the server's skip list into something the user can act on.
 *
 * Entries arrive as "report.docx: Unsupported file type: .docx". The filename
 * prefix is noise next to the row it is already displayed on, so it is trimmed
 * and the reason kept.
 */
function describeSkip(skipped: string[], filename: string): string {
  const mine = skipped.find((entry) => entry.startsWith(`${filename}:`)) ?? skipped[0];
  if (!mine) return 'No text extracted.';
  const reason = mine.slice(mine.indexOf(':') + 1).trim();
  return reason ? `${reason}.` : mine;
}

export function UploadPanel({ onIngested }: UploadPanelProps): ReactElement {
  const [items, setItems] = useState<UploadItem[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const dragDepth = useRef(0);

  const patch = useCallback((id: string, changes: Partial<UploadItem>): void => {
    setItems((current) =>
      current.map((item) => (item.id === id ? { ...item, ...changes } : item)),
    );
  }, []);

  const ingest = useCallback(
    async (files: File[]): Promise<void> => {
      if (files.length === 0) return;
      const jobs = files.map((file) => {
        const reason = rejectionReason(file);
        const item: UploadItem = {
          id: randomId('upload'),
          name: file.name,
          size: file.size,
          status: reason === null ? 'queued' : 'error',
          percent: 0,
          chunks: 0,
          error: reason,
        };
        return { file, item };
      });
      setItems((current) => [...jobs.map((job) => job.item), ...current].slice(0, 24));
      setBusy(true);

      // Sequential on purpose: chunking and embedding are CPU-bound on the
      // server, and one file at a time keeps the progress bars meaningful.
      for (const { file, item } of jobs) {
        if (item.status === 'error') continue;
        patch(item.id, { status: 'uploading' });
        try {
          const response = await uploadFile(file, (progress) => {
            patch(item.id, { percent: progress.percent });
          });
          const chunks = response.documents.reduce((total, doc) => total + doc.chunks, 0);
          if (response.documents.length === 0) {
            // The server says exactly why - "Unsupported file type: .docx", "No
            // extractable text" - and it is prefixed with the filename. Showing
            // a generic "skipped" instead leaves the user with nothing to act
            // on, which is the whole difference between a dead end and a fix.
            patch(item.id, {
              status: 'error',
              percent: 100,
              error: describeSkip(response.skipped, file.name),
            });
          } else {
            patch(item.id, { status: 'done', percent: 100, chunks, error: null });
          }
          onIngested(response);
        } catch (cause) {
          if (isAbort(cause)) {
            patch(item.id, { status: 'error', error: 'Cancelled.' });
          } else {
            patch(item.id, { status: 'error', error: describeError(cause) });
          }
        }
      }
      setBusy(false);
    },
    [onIngested, patch],
  );

  function handleFileInput(event: ChangeEvent<HTMLInputElement>): void {
    const list = event.target.files;
    const picked = list === null ? [] : Array.from(list);
    // Reset so re-picking the same file fires a change event again.
    event.target.value = '';
    void ingest(picked);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    void ingest(Array.from(event.dataTransfer.files));
  }

  function handleDragEnter(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    dragDepth.current += 1;
    setDragging(true);
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    // Child elements fire leave events too; only the outermost one counts.
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  }

  const totalChunks = items.reduce((total, item) => total + item.chunks, 0);

  return (
    <section className="panel upload" aria-labelledby="upload-heading">
      <div className="panel__header">
        <h2 id="upload-heading" className="panel__title">
          Documents
        </h2>
        {totalChunks > 0 && <span className="pill pill--quiet">{totalChunks} chunks added</span>}
      </div>

      <div
        className={`dropzone${dragging ? ' dropzone--active' : ''}`}
        onDragEnter={handleDragEnter}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <p className="dropzone__title">Drop files to index</p>
        <p className="dropzone__hint">{SUPPORTED_SUFFIXES.join(' · ')}</p>
        <button
          type="button"
          className="button button--small"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
        >
          {busy ? 'Uploading…' : 'Choose files'}
        </button>
        <input
          ref={inputRef}
          className="sr-only"
          type="file"
          multiple
          accept={SUPPORTED_SUFFIXES.join(',')}
          onChange={handleFileInput}
          tabIndex={-1}
        />
      </div>

      {items.length > 0 && (
        <ul className="upload__list">
          {items.map((item) => (
            <li key={item.id} className={`upload__item upload__item--${item.status}`}>
              <div className="upload__row">
                <span className="upload__name" title={item.name}>
                  {item.name}
                </span>
                <span className="upload__meta">
                  {item.status === 'done'
                    ? `${item.chunks} chunk${item.chunks === 1 ? '' : 's'}`
                    : formatSize(item.size)}
                </span>
              </div>
              {item.status === 'uploading' && (
                <div
                  className="progress"
                  role="progressbar"
                  aria-valuenow={item.percent}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-label={`Uploading ${item.name}`}
                >
                  <div className="progress__bar" style={{ width: `${item.percent}%` }} />
                </div>
              )}
              {item.error !== null && <p className="upload__error">{item.error}</p>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
