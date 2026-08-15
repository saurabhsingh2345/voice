"""The check that says a running process predates the code on disk.

Written after a `voice-web` served Hindi for a day through an engine module that
had been deleted underneath it. Every request failed, the checkout was correct,
and the product said nothing --- so the bug looked like a broken model.

The tests below are mostly about *not* warning. A staleness banner that fires
when nothing is wrong gets dismissed on sight, and a dismissed warning is worth
less than none, so the false-positive cases carry as much weight here as the
true one.
"""

from __future__ import annotations

from voiceagent.version import Snapshot, drift, newest_source, snapshot


def at(mtime, sha="abc1234", dirty=False, path="src/voiceagent/tts/router.py"):
    return Snapshot(sha=sha, dirty=dirty, newest_mtime=mtime, newest_path=path)


# --- the case it exists for ----------------------------------------------


def test_a_source_file_newer_than_the_process_is_reported():
    message = drift(at(1000.0), at(2000.0, sha="def5678", path="src/voiceagent/tts/x.py"))
    assert message is not None
    assert "src/voiceagent/tts/x.py" in message


def test_the_message_says_to_restart_because_that_is_the_only_fix():
    message = drift(at(1000.0), at(2000.0))
    assert "restart" in message.lower()


def test_the_message_names_both_shas_when_they_differ():
    message = drift(at(1000.0, sha="aaa1111"), at(2000.0, sha="bbb2222"))
    assert "aaa1111" in message and "bbb2222" in message


def test_a_dirty_tree_is_marked_in_the_label():
    assert at(1.0, sha="aaa1111", dirty=True).label() == "aaa1111+dirty"


# --- the false positives that would make it worthless ---------------------


def test_an_unchanged_tree_is_silent():
    assert drift(at(1000.0), at(1000.0)) is None


def test_a_commit_alone_does_not_warn():
    """The load-bearing case for choosing mtimes over SHA comparison.

    `git commit` moves HEAD without touching the working files, so the code a
    running process holds is still byte-identical to the tree. Warning here
    would fire after every commit --- which is to say, constantly, and then
    never read.
    """
    assert drift(at(1000.0, sha="aaa1111"), at(1000.0, sha="bbb2222")) is None


def test_an_older_mtime_does_not_warn():
    """Restoring an older file, or a clock stepping backwards, is not staleness.

    The comparison is deliberately `<=` rather than `!=`: only source that is
    *newer* than the running process can be code the process has not loaded.
    """
    assert drift(at(2000.0), at(1000.0)) is None


def test_a_bundle_with_no_checkout_never_warns():
    """No checkout means no watchable source, and nothing can be concluded.

    Code cannot change underneath a running `.app`, so silence is correct here
    rather than a degraded mode.
    """
    assert drift(Snapshot(), at(9999.0)) is None
    assert drift(at(1000.0), Snapshot()) is None


def test_a_checkout_without_git_still_detects_drift():
    """git is for the label; mtimes are for the decision. Losing git loses only
    the sha in the message, never the warning itself."""
    message = drift(at(1000.0, sha=None), at(2000.0, sha=None))
    assert message is not None
    assert "no git" not in message


# --- the real tree --------------------------------------------------------


def test_snapshot_of_this_checkout_is_watchable():
    now = snapshot()
    assert now.watchable
    assert now.newest_path.startswith("src/voiceagent")


def test_a_process_started_now_is_not_stale():
    """The end-to-end guard: taking a snapshot and immediately comparing it
    against the tree must be silent, or every server would cry wolf at boot."""
    assert drift(snapshot()) is None


def test_pycache_is_ignored():
    """Compiled caches are rewritten on import, so counting them would make
    every process report itself stale the moment it started."""
    from voiceagent import paths

    mtime, where = newest_source(paths.project_root())
    assert mtime is not None
    assert "__pycache__" not in where
