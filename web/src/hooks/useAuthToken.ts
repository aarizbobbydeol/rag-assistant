import { useSyncExternalStore } from 'react';

import { getToken, subscribeToken } from '../lib/token';

/**
 * Reads the bearer token as React state.
 *
 * The token is also cleared from inside the API client on any 401, so the sign-in
 * screen has to react to a change it did not initiate — hence an external store
 * rather than component state.
 */
export function useAuthToken(): string | null {
  return useSyncExternalStore(subscribeToken, getToken, getToken);
}
