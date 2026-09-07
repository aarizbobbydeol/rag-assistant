/**
 * The bearer token lives in a module-level variable and is mirrored into
 * localStorage.
 *
 * The in-memory copy is the one every request reads, so the app keeps working
 * in a browser where storage throws (private mode, blocked site data); the
 * mirror is what survives a reload.
 */

const STORAGE_KEY = 'rag.auth.token';

type Listener = (token: string | null) => void;

const listeners = new Set<Listener>();

function readStored(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

let cached: string | null = readStored();

export function getToken(): string | null {
  return cached;
}

export function setToken(token: string | null): void {
  if (cached === token) return;
  cached = token;
  try {
    if (token === null) {
      window.localStorage.removeItem(STORAGE_KEY);
    } else {
      window.localStorage.setItem(STORAGE_KEY, token);
    }
  } catch {
    // Storage is unavailable; the in-memory token still authorises this tab.
  }
  for (const listener of listeners) {
    listener(cached);
  }
}

export function subscribeToken(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
