# Web client

React 18 + TypeScript + Vite single-page app for the RAG assistant. No UI kit, no
state library, no router — three dependencies (`react`, `react-dom`, and the
Vite/TypeScript toolchain) and one hand-written stylesheet.

```bash
npm install
npm run dev        # http://localhost:5173, proxies the API to :8000
npm run build      # tsc --noEmit && vite build  ->  dist/
npm run typecheck
```

Run the API first (`uvicorn app.main:app --port 8000` from the repository root).
The dev server proxies `/api`, `/health` and `/metrics` to `http://localhost:8000`,
so the browser only ever talks to its own origin and there is no CORS story in
development. Change `API_TARGET` in `vite.config.ts` to point somewhere else.

For a production bundle served from a different origin than the API, copy
`.env.example` to `.env.local` and set `VITE_API_BASE_URL`.

## Layout

```
src/
  api/
    types.ts        TypeScript mirrors of the pydantic models in app/models.py
    client.ts       every HTTP call, bearer handling, ApiError
  lib/
    token.ts        in-memory + localStorage bearer token, with subscribers
    storage.ts      localStorage that survives blocked storage and a full quota
    settings.ts     top_k / mode / rerank / strict, persisted and validated
    conversations.ts session list and per-session transcript cache
    answer.ts       [n] marker parsing, quote location, score formatting
  hooks/
    useAuthToken.ts useSyncExternalStore over the token module
  components/
    ChatPanel       transcript, composer, empty and loading states
    AnswerBody      prose + clickable superscript citation pills
    SourcePanel     the panel behind a pill: identity, scores, quoted passage
    GroundednessBadge  per-answer support verdict, warning when ungrounded
    UploadPanel     drag-and-drop / picker, per-file progress and chunk counts
    SessionList     conversations, "new conversation", delete
    SettingsDrawer  retrieval knobs passed through on /api/chat
    LoginPanel      sign in / register
    ErrorBoundary   whole-tree crash guard with a retry
    Feedback        Spinner, EmptyState, ErrorNote
  App.tsx           composition and all cross-panel state
  styles.css        the entire design system, ~40 custom properties
```

## Behaviour worth knowing

**Citations are the point.** `AnswerBody` splits the answer on `[1]`, `[1,2]` and
`[1][2]`, renders each marker as a superscript pill, and clicking one opens
`SourcePanel` with the title, path, page, rank, retriever, every score that
contributed (final / dense / bm25 / rerank), the cited quote, and the full chunk
with the quote highlighted inside it. Retrieved-but-uncited passages are one
disclosure away, so it is visible when the model ignored a good hit. A marker the
backend could not resolve renders struck through and disabled rather than being
hidden — a visible dead citation is honest, a hidden one reads as an uncited
claim.

**Groundedness is always on screen.** Every answer carries a badge with the
support score and the supported/total sentence ratio, coloured by verdict and
expandable to the unsupported sentences, invalid markers and uncited count.
Abstained or low-support answers are styled as warnings, not statistics.

**Auth is optional until the API says otherwise.** The token is held in memory and
mirrored to localStorage, attached as `Authorization: Bearer` on every call, and
cleared by the client itself on any 401 — which flips the app to the sign-in
screen. With `RAG_AUTH_REQUIRED=false` (the dev default) the app never asks; the
header still offers a voluntary sign-in.

**Conversations survive a reload.** The sidebar list and the full answers —
citations, contexts and groundedness included — are cached in localStorage per
`session_id`. On a cold cache the client falls back to `GET /api/sessions/{id}`,
which returns the server transcript; those turns render as plain text because
`Turn` carries no citation data.

**Uploads are one request per file.** `fetch` cannot report request-body progress,
so `uploadFile` uses `XMLHttpRequest`; files go up sequentially, each with its own
progress bar and the chunk count from the response. Unsupported suffixes and
oversized files are rejected before the request.

## Assumptions about the API

The backend was written concurrently with this client. Where
`docs/ARCHITECTURE.md` did not pin a detail down, this is what the client does —
all of it is one constant or one function away from being changed.

| Assumption | Where |
|---|---|
| `POST /api/auth/token` takes JSON `{email, password}` and returns `{access_token, token_type, expires_in}` | `api/client.ts` |
| `POST /api/auth/register` takes JSON `{email, password}` and returns the user record | `api/client.ts` |
| `POST /api/ingest/upload` takes a repeated multipart field named `files` | `UPLOAD_FIELD` in `api/client.ts` |
| `POST /api/chat` returns `AnswerResult` verbatim | `api/types.ts` |
| `GET`/`DELETE /api/sessions/{id}` read and reset a transcript | `api/client.ts` |
| Health is at `/api/health`, falling back to `/health` on a 404 | `health()` in `api/client.ts` |

## Conventions

- `strict`, `noUnusedLocals`, `noUnusedParameters`, `noImplicitOverride` and
  `verbatimModuleSyntax` are all on; no `any` appears in an exported signature.
- Colour is defined once as custom properties on `:root` and redefined under
  `prefers-color-scheme: dark`, so no rule below the token block mentions a
  literal colour.
- The layout is a three-column grid that collapses to one column with an
  off-canvas sidebar and an overlaid source sheet below 900px.
