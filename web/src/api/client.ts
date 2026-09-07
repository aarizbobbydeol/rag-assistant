/**
 * The single place that talks to the FastAPI service.
 *
 * Every call attaches the bearer token, and a 401 anywhere clears it so the app
 * drops back to the sign-in screen instead of retrying with a dead credential.
 */

import { getToken, setToken } from '../lib/token';
import type {
  AnswerResult,
  ChatRequest,
  HealthResponse,
  IngestResponse,
  SearchRequest,
  SearchResponse,
  SessionResponse,
  TokenResponse,
  User,
} from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

/**
 * Multipart field name for `POST /api/ingest/upload`. The endpoint returns a
 * list of documents and a list of skipped files, so it takes a repeated field;
 * this client still posts one file per request in order to report per-file
 * progress.
 */
const UPLOAD_FIELD = 'files';

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: string | null;
  readonly traceId: string | null;

  constructor(
    status: number,
    message: string,
    code = 'error',
    detail: string | null = null,
    traceId: string | null = null,
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.detail = detail;
    this.traceId = traceId;
  }
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : null;
}

function asText(value: unknown): string | null {
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) return null;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/**
 * Turns a failed response into an `ApiError`, understanding both the app's
 * `ErrorResponse` shape and FastAPI's default `{"detail": ...}` envelope (whose
 * detail is an array for validation failures).
 */
function toApiError(status: number, payload: unknown, fallback: string): ApiError {
  const body = asRecord(payload);
  if (body === null) {
    return new ApiError(status, fallback);
  }
  const detail = asText(body.detail);
  const message = asText(body.error) ?? detail ?? fallback;
  const code = asText(body.error) ?? `http_${status}`;
  return new ApiError(status, message, code, detail, asText(body.trace_id));
}

async function readError(response: Response): Promise<ApiError> {
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    // Non-JSON body (a proxy error page, an empty 502); the status is the story.
  }
  return toApiError(response.status, payload, `${response.status} ${response.statusText}`.trim());
}

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  body?: unknown;
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  const token = getToken();
  if (token !== null) {
    headers.Authorization = `Bearer ${token}`;
  }

  let body: string | undefined;
  if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(options.body);
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: options.method ?? 'GET',
      headers,
      body,
      signal: options.signal,
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError(0, 'Cannot reach the API. Is the backend running?', 'network_error');
  }

  if (response.status === 401) {
    setToken(null);
    throw await readError(response);
  }
  if (!response.ok) {
    throw await readError(response);
  }
  if (response.status === 204) {
    return undefined as unknown as T;
  }
  return (await response.json()) as T;
}

// --------------------------------------------------------------------------- //
// Auth
// --------------------------------------------------------------------------- //
export async function login(email: string, password: string): Promise<TokenResponse> {
  const token = await request<TokenResponse>('/api/auth/token', {
    method: 'POST',
    body: { email, password },
  });
  setToken(token.access_token);
  return token;
}

export function register(email: string, password: string): Promise<User> {
  return request<User>('/api/auth/register', { method: 'POST', body: { email, password } });
}

export function logout(): void {
  setToken(null);
}

// --------------------------------------------------------------------------- //
// Service state
// --------------------------------------------------------------------------- //
export async function health(signal?: AbortSignal): Promise<HealthResponse> {
  try {
    return await request<HealthResponse>('/api/health', { signal });
  } catch (error) {
    // The health route is documented as unauthenticated; it is mounted either
    // under the API prefix or at the root depending on the router wiring.
    if (error instanceof ApiError && error.status === 404) {
      return request<HealthResponse>('/health', { signal });
    }
    throw error;
  }
}

// --------------------------------------------------------------------------- //
// Chat, search, sessions
// --------------------------------------------------------------------------- //
export function chat(payload: ChatRequest, signal?: AbortSignal): Promise<AnswerResult> {
  return request<AnswerResult>('/api/chat', { method: 'POST', body: payload, signal });
}

export function search(payload: SearchRequest, signal?: AbortSignal): Promise<SearchResponse> {
  return request<SearchResponse>('/api/search', { method: 'POST', body: payload, signal });
}

export function getSession(sessionId: string, signal?: AbortSignal): Promise<SessionResponse> {
  return request<SessionResponse>(`/api/sessions/${encodeURIComponent(sessionId)}`, { signal });
}

export function resetSession(sessionId: string): Promise<void> {
  return request<void>(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
}

// --------------------------------------------------------------------------- //
// Upload
// --------------------------------------------------------------------------- //
export interface UploadProgress {
  loaded: number;
  total: number;
  /** 0-100, clamped; 0 while the total is still unknown. */
  percent: number;
}

/**
 * Uploads one file with progress.
 *
 * `fetch` cannot report request-body progress, so this is the one call that
 * uses `XMLHttpRequest`.
 */
export function uploadFile(
  file: File,
  onProgress?: (progress: UploadProgress) => void,
  signal?: AbortSignal,
): Promise<IngestResponse> {
  return new Promise<IngestResponse>((resolve, reject) => {
    const form = new FormData();
    form.append(UPLOAD_FIELD, file, file.name);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/api/ingest/upload`, true);
    xhr.responseType = 'text';
    xhr.setRequestHeader('Accept', 'application/json');
    const token = getToken();
    if (token !== null) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    }

    const abort = (): void => xhr.abort();
    if (signal) {
      if (signal.aborted) {
        reject(new DOMException('Upload aborted', 'AbortError'));
        return;
      }
      signal.addEventListener('abort', abort, { once: true });
    }
    const cleanup = (): void => {
      if (signal) signal.removeEventListener('abort', abort);
    };

    if (onProgress) {
      xhr.upload.onprogress = (event): void => {
        const total = event.lengthComputable ? event.total : file.size;
        const percent = total > 0 ? Math.min(100, Math.round((event.loaded / total) * 100)) : 0;
        onProgress({ loaded: event.loaded, total, percent });
      };
    }

    xhr.onload = (): void => {
      cleanup();
      let payload: unknown = null;
      try {
        payload = JSON.parse(xhr.responseText) as unknown;
      } catch {
        // Handled below: a non-JSON body on a 2xx is an error either way.
      }
      if (xhr.status === 401) {
        setToken(null);
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        if (payload === null) {
          reject(new ApiError(xhr.status, 'Upload succeeded but the response was unreadable.'));
          return;
        }
        onProgress?.({ loaded: file.size, total: file.size, percent: 100 });
        resolve(payload as IngestResponse);
        return;
      }
      reject(toApiError(xhr.status, payload, `Upload failed (${xhr.status}).`));
    };

    xhr.onerror = (): void => {
      cleanup();
      reject(new ApiError(0, 'Cannot reach the API. Is the backend running?', 'network_error'));
    };
    xhr.ontimeout = (): void => {
      cleanup();
      reject(new ApiError(0, 'The upload timed out.', 'timeout'));
    };
    xhr.onabort = (): void => {
      cleanup();
      reject(new DOMException('Upload aborted', 'AbortError'));
    };

    xhr.send(form);
  });
}

/** Human-readable message for anything thrown by this module. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.detail !== null && error.detail !== error.message) {
      return `${error.message} — ${error.detail}`;
    }
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return 'Something went wrong.';
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}
