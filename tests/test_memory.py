"""Tests for the conversation store: ordering, LRU, TTL and concurrency."""

from __future__ import annotations

import threading

from app.memory import SessionStore
from app.models import Turn


def turn(content: str, role: str = "user") -> Turn:
    return Turn(role=role, content=content)


def test_history_returns_last_turns_oldest_first() -> None:
    store = SessionStore(max_sessions=8, ttl_s=600)
    for i in range(5):
        store.append("s1", turn(f"q{i}"))

    assert [t.content for t in store.history("s1", 3)] == ["q2", "q3", "q4"]
    assert [t.content for t in store.history("s1", 99)] == ["q0", "q1", "q2", "q3", "q4"]


def test_history_tolerates_unknown_and_missing_sessions() -> None:
    store = SessionStore()
    store.append("s1", turn("hello"))

    assert store.history("nope", 5) == []
    assert store.history(None, 5) == []
    assert store.history("", 5) == []
    assert store.history("s1", 0) == []
    assert store.history("s1", -3) == []


def test_append_without_session_id_is_a_noop() -> None:
    store = SessionStore()
    store.append(None, turn("stateless"))
    store.append("", turn("stateless"))

    assert len(store) == 0


def test_history_is_a_copy_so_callers_cannot_corrupt_the_store() -> None:
    store = SessionStore()
    store.append("s1", turn("original"))

    fetched = store.history("s1", 1)
    fetched[0].content = "mutated"
    fetched.clear()

    assert [t.content for t in store.history("s1", 1)] == ["original"]


def test_appended_turn_is_snapshotted() -> None:
    store = SessionStore()
    t = turn("before")
    store.append("s1", t)
    t.content = "after"

    assert [x.content for x in store.history("s1", 1)] == ["before"]


def test_reset_drops_only_the_named_session() -> None:
    store = SessionStore()
    store.append("s1", turn("a"))
    store.append("s2", turn("b"))

    assert store.reset("s1") is True
    assert store.reset("s1") is False
    assert store.reset(None) is False
    assert store.history("s1", 5) == []
    assert [t.content for t in store.history("s2", 5)] == ["b"]


def test_lru_eviction_respects_reads_as_touches() -> None:
    store = SessionStore(max_sessions=2, ttl_s=600)
    store.append("a", turn("a"))
    store.append("b", turn("b"))
    store.history("a", 1)  # makes "b" the least recently used
    store.append("c", turn("c"))

    assert len(store) == 2
    assert store.history("b", 5) == []
    assert [t.content for t in store.history("a", 5)] == ["a"]
    assert [t.content for t in store.history("c", 5)] == ["c"]


def test_purge_expired_removes_idle_sessions_only() -> None:
    store = SessionStore(max_sessions=10, ttl_s=60)
    store.append("old", turn("old"))
    store.append("fresh", turn("fresh"))
    store._sessions["old"].touched_at -= 3600  # pretend it went idle an hour ago

    assert store.purge_expired() == 1
    assert store.purge_expired() == 0
    assert len(store) == 1
    assert [t.content for t in store.history("fresh", 5)] == ["fresh"]


def test_reads_and_writes_purge_lazily() -> None:
    store = SessionStore(max_sessions=10, ttl_s=60)
    store.append("old", turn("old"))
    store._sessions["old"].touched_at -= 3600

    assert store.history("old", 5) == []
    assert len(store) == 0


def test_non_positive_ttl_disables_expiry() -> None:
    store = SessionStore(max_sessions=10, ttl_s=0)
    store.append("s1", turn("a"))
    store._sessions["s1"].touched_at -= 10_000

    assert store.purge_expired() == 0
    assert [t.content for t in store.history("s1", 5)] == ["a"]


def test_concurrent_appends_do_not_lose_or_mix_turns() -> None:
    store = SessionStore(max_sessions=64, ttl_s=600)
    threads = 8
    per_thread = 100
    errors: list[BaseException] = []
    start = threading.Barrier(threads)

    def hammer(worker: int) -> None:
        try:
            start.wait(timeout=10)
            for i in range(per_thread):
                store.append("shared", turn(f"w{worker}-{i}"))
                store.append(f"own-{worker}", turn(f"w{worker}-{i}"))
                store.history("shared", 5)
                store.purge_expired()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            errors.append(exc)

    workers = [threading.Thread(target=hammer, args=(w,)) for w in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=30)

    assert not errors
    assert not any(w.is_alive() for w in workers)

    shared = store.history("shared", 10_000)
    assert len(shared) == threads * per_thread
    assert {t.content for t in shared} == {
        f"w{w}-{i}" for w in range(threads) for i in range(per_thread)
    }
    for worker in range(threads):
        own = store.history(f"own-{worker}", 10_000)
        assert [t.content for t in own] == [f"w{worker}-{i}" for i in range(per_thread)]


def test_concurrent_writers_respect_the_session_cap() -> None:
    store = SessionStore(max_sessions=5, ttl_s=600)
    errors: list[BaseException] = []

    def hammer(worker: int) -> None:
        try:
            for i in range(200):
                store.append(f"s-{worker}-{i}", turn("x"))
                store.reset(f"s-{worker}-{i - 3}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    workers = [threading.Thread(target=hammer, args=(w,)) for w in range(6)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=30)

    assert not errors
    assert len(store) <= 5
