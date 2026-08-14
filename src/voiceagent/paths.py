"""Where this app keeps its data, in a checkout and in a bundle.

Every store in the project used to resolve its own location the same way:

    Path(__file__).resolve().parents[3] / "data"

which is the project root when the package is imported from `src/`, and is
`site-packages/data` when it is installed. In a `.app` that lands inside
`Contents/Resources/runtime/lib/python3.12/site-packages/data` --- inside the
application bundle. Two things are wrong with that and both are quiet:

  * A bundle is read-only in any normal install (and signed bundles genuinely
    are). Enrolling a voice fails, or worse, half-succeeds.
  * Anything that did get written is *inside the app*, so it disappears the next
    time the app is replaced. A consented voice recording is not a cache.

`data_dir()` gives one answer for both cases, and the checkout keeps behaving
exactly as it did so nothing about local development changes.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Overrides everything. The desktop shell sets this, because a launcher that
#: knows where it put things beats a heuristic that has to guess.
ENV_VAR = "VOICEAGENT_DATA_DIR"

#: Where a packaged app writes on macOS. Not `~/.voiceagent`: this is the
#: location the platform backs up, migrates between machines, and shows in
#: Storage settings, and voice profiles are exactly the kind of data a user
#: expects to survive a laptop change.
MACOS_APP_SUPPORT = "Library/Application Support/Local Voice Agent"


def project_root() -> Path | None:
    """The checkout this package was imported from, if it is one.

    Detected by `pyproject.toml` rather than by looking for `src/`, because an
    editable install can put the package almost anywhere.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def data_dir() -> Path:
    """The writable root for voices, clips, history and downloaded checkpoints.

    Order is deliberate: an explicit setting, then the checkout, then the
    platform location. The checkout comes before the platform directory so that
    running from source keeps using `./data` --- otherwise every existing
    profile on a developer's machine would silently disappear from the UI the
    first time this landed.
    """
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()

    root = project_root()
    if root is not None:
        return root / "data"

    return Path.home() / MACOS_APP_SUPPORT


def ensure(*parts: str) -> Path:
    """`data_dir()` joined with `parts`, created if missing."""
    path = data_dir().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
