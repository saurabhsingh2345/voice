"""Where the app puts data, and which port it listens on.

Both were fine in a checkout and wrong in a bundle, which is the pattern for
everything packaging turned up: the failure only appears once the code is
somewhere other than the developer's tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voiceagent import paths
from voiceagent.web.server import DEFAULT_PORT, PORT_ENV, resolved_port


# --- data location ----------------------------------------------------------


def test_the_env_override_wins(monkeypatch, tmp_path):
    """The desktop shell sets this. A launcher that knows where it put things
    beats a heuristic that has to guess."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "elsewhere"))
    assert paths.data_dir() == tmp_path / "elsewhere"


def test_a_checkout_keeps_using_its_own_data_dir(monkeypatch):
    """Ordered before the platform directory on purpose: otherwise every profile
    on a developer's machine would silently vanish from the UI the first time
    this landed."""
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    root = paths.project_root()
    assert root is not None, "these tests run from a checkout"
    assert paths.data_dir() == root / "data"


def test_outside_a_checkout_it_falls_back_to_the_platform_directory(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.setattr(paths, "project_root", lambda: None)
    assert paths.data_dir() == Path.home() / paths.MACOS_APP_SUPPORT


def test_it_never_resolves_inside_an_installed_package(monkeypatch):
    """The bug this module exists for. Every store used to compute
    `Path(__file__).parents[3] / "data"`, which is the project root from a
    checkout and `site-packages/data` when installed -- inside Contents/Resources
    in a .app. That is read-only in a normal install and replaced wholesale on
    update, and a consented voice recording is not a cache.
    """
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.setattr(paths, "project_root", lambda: None)
    resolved = str(paths.data_dir())
    assert "site-packages" not in resolved
    assert ".app/Contents" not in resolved


def test_the_stores_go_through_it():
    """Asserted on the modules, because a store that reintroduces its own
    `__file__` arithmetic would pass every other test here."""
    import inspect

    from voiceagent.storage import db
    from voiceagent.voice_clone import store

    for module in (db, store):
        source = inspect.getsource(module)
        assert 'parents[3] / "data"' not in source, f"{module.__name__} resolves its own path"
        assert "paths.data_dir()" in source


def test_ensure_creates_the_directory(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "fresh"))
    made = paths.ensure("voices", "abc")
    assert made.is_dir()


# --- port -------------------------------------------------------------------


def test_the_default_port_is_what_the_shell_looks_for(monkeypatch):
    monkeypatch.delenv(PORT_ENV, raising=False)
    assert resolved_port() == DEFAULT_PORT == 8823


def test_the_port_can_be_overridden(monkeypatch):
    """Needed because a second instance -- the packaged app while a checkout is
    already serving -- otherwise dies on "address already in use". That happened
    during bundle testing, and worse: a test appeared to pass because the 200s
    were coming from an unrelated server on the same port. A silent wrong pass
    beats a crash for damage done."""
    monkeypatch.setenv(PORT_ENV, "8824")
    assert resolved_port() == 8824


@pytest.mark.parametrize("bad", ["nope", "0", "70000", "-1"])
def test_a_bad_port_fails_loudly(monkeypatch, bad):
    monkeypatch.setenv(PORT_ENV, bad)
    with pytest.raises(SystemExit):
        resolved_port()


def test_the_host_is_not_configurable():
    """Deliberate. Making it an environment variable would turn one setting into
    the difference between a private assistant and one answering the network."""
    import inspect

    from voiceagent.web import server

    source = inspect.getsource(server.main)
    assert '"127.0.0.1"' in source
    assert "HOST_ENV" not in dir(server)
