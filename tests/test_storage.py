"""Tests for encrypted history and memory.

The claims worth testing are the privacy ones: content is unreadable on disk,
retrieval finds the right thing, and deletion actually destroys rather than
merely unlinks.
"""

from __future__ import annotations

import pytest

from voiceagent.storage.db import EncryptedStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store with the Keychain stubbed, so tests never touch the real one."""
    slot: dict[str, str] = {}
    import keyring

    monkeypatch.setattr(keyring, "get_password", lambda s, u: slot.get(f"{s}/{u}"))
    monkeypatch.setattr(keyring, "set_password", lambda s, u, v: slot.__setitem__(f"{s}/{u}", v))
    monkeypatch.setattr(keyring, "delete_password", lambda s, u: slot.pop(f"{s}/{u}", None))

    s = EncryptedStore(tmp_path / "history.db")
    yield s
    s.close()


SECRET = "my bank password is hunter2 and I live at 14 Elm Street"


def test_roundtrip(store):
    convo = store.start_conversation("test")
    store.add_message(convo, "user", "hello there")
    store.add_message(convo, "assistant", "hi back")

    messages = store.messages(convo)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert [m.content for m in messages] == ["hello there", "hi back"]


def test_content_is_not_readable_on_disk(store):
    convo = store.start_conversation()
    store.add_message(convo, "user", SECRET)
    store.close()

    raw = (store.path).read_bytes()
    assert b"hunter2" not in raw
    assert b"Elm Street" not in raw
    # The schema is NOT hidden -- documented limitation vs SQLCipher.
    assert b"messages" in raw


def test_history_survives_reopen(store, tmp_path, monkeypatch):
    convo = store.start_conversation()
    store.add_message(convo, "user", "persisted across restarts")
    store.close()

    reopened = EncryptedStore(store.path)
    assert reopened.messages(convo)[0].content == "persisted across restarts"
    reopened.close()


# --- retrieval ------------------------------------------------------------


def test_recall_finds_the_relevant_memory(store):
    store.remember("The user's dog is called Biscuit")
    store.remember("The user works as a structural engineer in Leeds")
    store.remember("The user prefers tea over coffee in the morning")

    hits = store.recall("what is my dog called?", limit=1)
    assert len(hits) == 1
    assert "Biscuit" in hits[0].content


def test_recall_returns_nothing_for_unrelated_queries(store):
    store.remember("The user's dog is called Biscuit")
    assert store.recall("quantum chromodynamics lattice gauge") == []


def test_recall_ignores_stopword_only_queries(store):
    store.remember("The user's dog is called Biscuit")
    assert store.recall("what is the that a an") == []


def test_recall_prefers_the_more_precise_memory(store):
    """A short exact memory should beat a long one that merely mentions the word."""
    store.remember("Dog: Biscuit")
    store.remember(
        "Long rambling note that mentions a dog once but is mostly about "
        "unrelated matters such as invoices, deadlines, parking permits, "
        "kitchen renovations and a great many other tedious subjects"
    )
    hits = store.recall("dog", limit=1)
    assert hits[0].content == "Dog: Biscuit"


def test_recall_respects_limit(store):
    for i in range(10):
        store.remember(f"memory number {i} about scheduling")
    assert len(store.recall("scheduling", limit=3)) == 3


# --- deletion -------------------------------------------------------------


def test_delete_all_removes_everything_and_the_key(store):
    convo = store.start_conversation()
    store.add_message(convo, "user", SECRET)
    store.remember("something private")

    counts = store.delete_all()
    assert counts == {"messages": 1, "conversations": 1, "memories": 1}
    assert not store.path.exists()

    import keyring
    from voiceagent.storage.db import KEYRING_SERVICE, KEYRING_USER

    assert keyring.get_password(KEYRING_SERVICE, KEYRING_USER) is None


def test_data_written_after_wipe_uses_a_fresh_key(store):
    convo = store.start_conversation()
    store.add_message(convo, "user", SECRET)
    store.delete_all()

    new_convo = store.start_conversation()
    store.add_message(new_convo, "user", "after the wipe")
    assert store.messages(new_convo)[0].content == "after the wipe"


def test_rows_encrypted_with_a_destroyed_key_are_unreadable(store, tmp_path):
    """If ciphertext somehow survives, it must not be decryptable."""
    convo = store.start_conversation()
    store.add_message(convo, "user", SECRET)

    conn = store.connect()
    orphan = conn.execute("SELECT content_enc FROM messages").fetchone()[0]
    store.delete_all()

    fresh = EncryptedStore(tmp_path / "history.db")
    fresh.start_conversation()  # forces a new key
    assert "unreadable" in fresh._decrypt(orphan)
    fresh.close()


# --- agent integration ----------------------------------------------------


class _NullEngine:
    name = "null"
    def load(self): ...
    def unload(self): ...
    async def stream(self, *a, **k):
        from voiceagent.llm.base import Chunk
        yield Chunk(text="ok", time_to_first_token_ms=1.0)
    @property
    def resident_bytes(self): return 0


async def test_agent_persists_both_sides_of_a_turn(store):
    from voiceagent.llm.agent import Agent

    agent = Agent(_NullEngine(), store=store)
    async for _ in agent.turn("remember I like tea"):
        pass

    saved = store.messages(agent.conversation_id)
    assert [m.role for m in saved] == ["user", "assistant"]
    assert saved[0].content == "remember I like tea"


async def test_agent_injects_relevant_memories(store):
    from voiceagent.llm.agent import Agent

    store.remember("The user's dog is called Biscuit")
    store.remember("The user dislikes coriander")

    agent = Agent(_NullEngine(), store=store)
    async for _ in agent.turn("what is my dog called?"):
        pass

    injected = [m for m in agent.history if m.role == "system" and "Biscuit" in m.content]
    assert injected, "relevant memory was not injected into the prompt"
    # The irrelevant one must not be dragged in.
    assert "coriander" not in injected[0].content


async def test_agent_injects_nothing_when_nothing_matches(store):
    from voiceagent.llm.agent import Agent

    store.remember("The user's dog is called Biscuit")
    agent = Agent(_NullEngine(), store=store)
    before = len(agent.history)
    async for _ in agent.turn("explain lattice gauge theory"):
        pass
    system_msgs = [m for m in agent.history[before:] if m.role == "system"]
    assert system_msgs == []


async def test_agent_without_a_store_still_works(store):
    from voiceagent.llm.agent import Agent

    agent = Agent(_NullEngine())
    async for _ in agent.turn("hello"):
        pass
    assert agent.conversation_id is None
