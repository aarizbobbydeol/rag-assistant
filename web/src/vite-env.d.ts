/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute API origin, no trailing slash. Empty means "same origin". */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
