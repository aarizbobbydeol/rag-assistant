"""Prompt construction for answering, condensing and LLM reranking.

The context block format defined here is a contract in three directions: the
prompt tells the model to cite `[n]`, `app.generation.citations` resolves those
markers *positionally* against the list this module returns, and
`app.generation.llm.ExtractiveLLM` parses the very same rendering back out of
the user message. So the numbering, the markers and the returned list must
always describe exactly the same passages - a context that is trimmed away for
budget must leave no trace in the text.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.config import Settings
from app.models import LLMMessage, ScoredChunk, Turn
from app.utils import estimate_tokens, normalize_whitespace, truncate

# Labels the extractive backend keys off when it parses a prompt back apart.
CONTEXT_HEADER = "Context passages:"
QUESTION_PREFIX = "Question:"
NO_CONTEXT = "(no context passages were retrieved)"

# The guard rails compare an answer against ``settings.abstain_message`` to
# decide whether the model abstained, so the sentence baked into the prompt is
# read off the same field default instead of being duplicated as a literal.
ABSTENTION_SENTENCE: str = str(Settings.model_fields["abstain_message"].default)

SYSTEM_PROMPT = f"""You are a retrieval-augmented research assistant. You answer questions \
using only the numbered context passages supplied with each question.

Rules:
1. Ground every claim in the context. Never use prior knowledge, never guess, and never \
fill a gap with a plausible-sounding detail. If the passages disagree, say so and cite both.
2. Put a citation marker on every factual sentence, written as the passage number in square \
brackets immediately before the closing period, like this: Refunds take five business days [2].
3. Cite only passage numbers that appear in the context you were given. Inventing a number, \
or citing a passage that does not actually support the sentence, is the worst failure mode \
there is - it looks correct and is not.
4. A sentence may cite more than one passage: write [1][3] when two passages jointly support it.
5. Answer in three sentences or fewer unless the question genuinely needs more. Prefer the \
words of the source over your own paraphrase, and do not add a preamble such as \
"According to the context".
6. If the passages do not contain the answer, reply with exactly this sentence and nothing \
else: {ABSTENTION_SENTENCE}"""

_CONDENSE_SYSTEM = """You rewrite a follow-up question into a standalone question for a \
search engine that cannot see the conversation.

Resolve pronouns and elliptical references ("it", "that one", "what about later?") against \
the transcript, keep every entity, constraint and qualifier from the original wording, and \
add nothing that was not asked. If the question already stands on its own, repeat it \
unchanged. Reply with the rewritten question only - no quotes, no explanation."""

_RERANK_SYSTEM = """You score how well each numbered passage answers a search query.

Judge only whether the passage contains information that answers the query; ignore writing \
quality, length and how interesting it is. Reply with a JSON array of objects sorted by \
descending score, each shaped {"index": <passage number>, "score": <0-10>}, covering every \
passage exactly once. Output the JSON array and nothing else."""

_RERANK_SNIPPET_CHARS = 600


def _render_block(marker: int, context: ScoredChunk) -> str:
    """One context passage: `[n] Title (p.3) :: source` followed by its text."""
    chunk = context.chunk
    return f"[{marker}] {chunk.locator()} :: {chunk.source}\n{chunk.text.strip()}"


def _shrink_to_fit(marker: int, context: ScoredChunk, max_tokens: int) -> tuple[str, bool]:
    """Truncate a single passage until its block fits ``max_tokens``.

    Only used for the first passage: returning an empty context guarantees an
    abstention, which is a worse answer than one built on the best passage cut
    short, so the head of the highest-ranked chunk is kept rather than dropped.
    """
    chunk = context.chunk
    text = chunk.text.strip()
    header = f"[{marker}] {chunk.locator()} :: {chunk.source}\n"
    limit = len(text)
    for _ in range(12):
        block = header + truncate(text, limit)
        if estimate_tokens(block) <= max_tokens:
            return block, True
        limit = int(limit * 0.7)
        if limit <= 16:
            break
    return "", False


def format_context(
    contexts: Sequence[ScoredChunk], max_tokens: int
) -> tuple[str, list[ScoredChunk]]:
    """Render passages as numbered blocks within a token budget.

    Returns the rendered text and the passages it actually contains, in the same
    order, so that marker `n` always refers to element `n - 1` of the returned
    list. Trimming stops at the first passage that does not fit rather than
    skipping it, which keeps the surviving passages the top-ranked prefix of the
    input.
    """
    if not contexts or max_tokens <= 0:
        return "", []

    blocks: list[str] = []
    included: list[ScoredChunk] = []
    for context in contexts:
        marker = len(included) + 1
        candidate = "\n\n".join([*blocks, _render_block(marker, context)])
        if estimate_tokens(candidate) > max_tokens:
            if included:
                break
            block, fitted = _shrink_to_fit(marker, context, max_tokens)
            if not fitted:
                break
            blocks.append(block)
            included.append(context)
            break
        blocks.append(_render_block(marker, context))
        included.append(context)

    return "\n\n".join(blocks), included


def build_answer_messages(
    question: str,
    contexts: Sequence[ScoredChunk],
    history: Sequence[Turn] = (),
    max_context_tokens: int = 6000,
) -> list[LLMMessage]:
    """System prompt, prior turns, then the context + question message.

    Prior turns are replayed as real user/assistant messages rather than being
    flattened into the final prompt: chat models weight their own past replies
    differently from quoted text, and it keeps the context block unambiguous for
    the extractive backend, which reads only the last user message. The caller
    decides how many turns to pass (``settings.history_turns``); everything
    given is folded in.
    """
    rendered, _ = format_context(contexts, max_context_tokens)
    messages = [LLMMessage(role="system", content=SYSTEM_PROMPT)]
    messages.extend(LLMMessage(role=turn.role, content=turn.content) for turn in history)
    messages.append(
        LLMMessage(
            role="user",
            content=(
                f"{CONTEXT_HEADER}\n{rendered or NO_CONTEXT}\n\n"
                f"{QUESTION_PREFIX} {question.strip()}"
            ),
        )
    )
    return messages


def build_condense_messages(question: str, history: Sequence[Turn] = ()) -> list[LLMMessage]:
    """Rewrite a follow-up into a standalone question.

    The transcript is quoted inside the user message instead of being replayed
    as turns, because the model's job here is to *read* the conversation, not to
    continue it.
    """
    if not history:
        return [
            LLMMessage(role="system", content=_CONDENSE_SYSTEM),
            LLMMessage(role="user", content=f"Follow-up question: {question.strip()}"),
        ]
    transcript = "\n".join(
        f"{'User' if turn.role == 'user' else 'Assistant'}: {normalize_whitespace(turn.content)}"
        for turn in history
    )
    return [
        LLMMessage(role="system", content=_CONDENSE_SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"Conversation so far:\n{transcript}\n\n"
                f"Follow-up question: {question.strip()}\n\nStandalone question:"
            ),
        ),
    ]


def build_rerank_messages(query: str, candidates: Sequence[ScoredChunk]) -> list[LLMMessage]:
    """Ask for relevance scores over numbered candidates.

    Passages are numbered from 1 in candidate order and truncated, since a
    reranker only needs enough text to judge topicality and the candidate list
    is several times longer than the final context window.
    """
    blocks = [
        f"[{i}] {candidate.chunk.locator()}\n"
        f"{truncate(candidate.chunk.text.strip(), _RERANK_SNIPPET_CHARS)}"
        for i, candidate in enumerate(candidates, start=1)
    ]
    body = "\n\n".join(blocks) if blocks else NO_CONTEXT
    return [
        LLMMessage(role="system", content=_RERANK_SYSTEM),
        LLMMessage(role="user", content=f"Query: {query.strip()}\n\nPassages:\n{body}"),
    ]
