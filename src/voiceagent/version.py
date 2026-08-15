"""What code this process is running, and whether the checkout has moved on.

The failure this exists for: a `voice-web` left running from the previous
afternoon kept serving Hindi through `tts/indic_engine.py` for a day after that
module was deleted and `f5-tts` uninstalled beneath it. Every Hindi request
failed, the code on disk was correct, and nothing in the product said why.
Python binds modules at import, so a long-lived server quietly becomes a museum
of whatever the checkout looked like when it started.

**mtimes decide; the SHA is only for the human reading the message.** That is
the whole design, and it is worth the paragraph:

  * File mtimes catch uncommitted edits, which is the ordinary case while
    developing and the case a SHA comparison is blind to.
  * A commit does not change a running process. Warning on SHA drift alone would
    fire after every `git commit` even though the loaded code is byte-identical
    to the tree. A warning that cries wolf gets ignored, and that is the one
    outcome worse than no warning at all.

Limits, stated because they are the difference between a check you can trust and
one you think you can:

  * This watches **this project's source only**. A `uv sync` that changes
    installed dependencies without touching `src/` goes unnoticed --- and a
    removed dependency was half of what broke the server that prompted this.
    Restart after `uv sync` regardless of what the banner says.
  * In a packaged `.app` there is no checkout and usually no `git`, so every
    function here returns `None` and callers show nothing. That is correct
    rather than degraded: code cannot change underneath a running bundle.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from voiceagent import paths

#: Where the project's own importable source lives, relative to the checkout.
#: Only this subtree is watched --- `tests/` and `eval_out/` change constantly
#: and none of it is imported into a running server.
SOURCE_SUBDIR = "src/voiceagent"

#: `git` is consulted for a human-readable label, never for the staleness
#: decision, so a slow or missing git must not be able to hang a request.
GIT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class Snapshot:
    """The state of the source tree at one instant.

    Every field is optional because all of them are unavailable in a bundle,
    and two of them are unavailable in a checkout with no `git`.
    """

    #: Short HEAD SHA, or None outside a git repository.
    sha: str | None = None
    #: Whether the working tree had uncommitted changes. None if unknown.
    dirty: bool | None = None
    #: Newest mtime across the watched source, or None if there is no checkout.
    newest_mtime: float | None = None
    #: Path of the file holding that mtime, relative to the checkout. For the
    #: message --- naming the file that changed is what makes the warning
    #: actionable rather than merely alarming.
    newest_path: str | None = None

    @property
    def watchable(self) -> bool:
        """Whether this snapshot can support a staleness comparison at all."""
        return self.newest_mtime is not None

    def label(self) -> str:
        """A short human-readable identity, for logs and the UI."""
        if self.sha is None:
            return "no git"
        return f"{self.sha}{'+dirty' if self.dirty else ''}"


def _git(root: Path, *args: str) -> str | None:
    """Run a git command in `root`, returning stripped stdout or None.

    Returns None for every failure mode --- git absent, not a repository, a
    timeout --- because none of them is an error here. The caller degrades to a
    label rather than an exception.
    """
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def head_sha(root: Path) -> tuple[str | None, bool | None]:
    """`(short_sha, dirty)` for the checkout at `root`."""
    sha = _git(root, "rev-parse", "--short", "HEAD")
    if sha is None:
        return None, None
    #: `--porcelain` prints one line per changed path and nothing at all when
    #: the tree is clean, so emptiness *is* the answer.
    status = _git(root, "status", "--porcelain")
    return sha, bool(status)


def newest_source(root: Path) -> tuple[float | None, str | None]:
    """The newest mtime under the watched source subtree, and whose it is.

    Ties go to whichever path `rglob` reaches first; when two files share an
    mtime either name is equally true and the message only needs one.
    """
    source = root / SOURCE_SUBDIR
    if not source.is_dir():
        return None, None

    newest: float | None = None
    where: str | None = None
    for path in source.rglob("*.py"):
        #: Compiled caches are rewritten on import, so including them would make
        #: every process look stale the moment it started.
        if "__pycache__" in path.parts:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            #: A file deleted mid-walk is exactly the drift being watched for;
            #: skipping it here is fine because the deletion moved its parent
            #: directory's mtime and the next scan sees a different tree anyway.
            continue
        if newest is None or mtime > newest:
            newest = mtime
            where = str(path.relative_to(root))
    return newest, where


def snapshot(*, with_git: bool = True) -> Snapshot:
    """Capture the current source state.

    `with_git=False` skips the subprocesses. Callers polling this on a request
    path use it to stay cheap: mtimes alone decide staleness, so the SHA is only
    worth fetching when there is something to report.
    """
    root = paths.project_root()
    if root is None:
        return Snapshot()

    mtime, where = newest_source(root)
    sha, dirty = head_sha(root) if with_git else (None, None)
    return Snapshot(sha=sha, dirty=dirty, newest_mtime=mtime, newest_path=where)


def drift(started: Snapshot, current: Snapshot | None = None) -> str | None:
    """Describe how the tree has moved since `started`, or None if it has not.

    Returns a message written to be read by whoever is about to lose an hour to
    it, so it names the file and says what to do.
    """
    if not started.watchable:
        #: No checkout, or the source subtree was not found. Nothing can be
        #: concluded, and a warning that cannot be acted on is noise.
        return None

    current = current or snapshot()
    if not current.watchable:
        return None

    assert started.newest_mtime is not None and current.newest_mtime is not None
    if current.newest_mtime <= started.newest_mtime:
        return None

    #: Only now is a SHA worth a subprocess. Callers on a request path pass a
    #: `with_git=False` snapshot to stay cheap, so the label is filled in here,
    #: on the rare branch that actually prints one.
    if current.sha is None:
        root = paths.project_root()
        if root is not None:
            sha, dirty = head_sha(root)
            current = Snapshot(
                sha=sha,
                dirty=dirty,
                newest_mtime=current.newest_mtime,
                newest_path=current.newest_path,
            )

    changed = current.newest_path or "a source file"
    detail = ""
    if current.sha and started.sha and current.sha != started.sha:
        detail = f" HEAD is now {current.label()}, this process loaded {started.label()}."
    elif started.sha:
        detail = f" This process loaded {started.label()}."

    return (
        f"This server is running code from before {changed} changed on disk."
        f"{detail} Python binds modules at import, so restarting is the only way "
        "to pick the change up."
    )
