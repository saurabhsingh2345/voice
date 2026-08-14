"""Register `voiceagent.text.numbers` under the name `num2words`.

`misaki.en` does `from num2words import num2words` at module scope, so the real
LGPL package has to be importable before misaki is — or something has to be
there under that name. This puts our own implementation there.

Same technique as `tts.kokoro_engine._disable_gpl_espeak_fallback`, and for the
same reason: a licence obligation that only bites when Python is frozen into one
opaque binary, which is exactly what the desktop bundle does. The difference is
that the espeak stub deliberately *fails* (mlx-audio handles that by disabling
the fallback), whereas this one has to actually work — numbers are common in
speech and skipping them is not an option.

Deliberately not a vendored copy of num2words. Vendoring LGPL code carries the
same relink obligation into the bundle; a clean-room implementation with no
derived source does not. `voiceagent.text.numbers` was written against the
library's *observed output* rather than its source, and the parity tests record
that.

If the real package happens to be installed, this still wins: `install()` runs
before misaki is imported and `sys.modules` short-circuits the import. That is
intentional — it means the same code path is exercised whether or not the
package is present, so tests cannot pass here and fail in a bundle.
"""

from __future__ import annotations

import sys
import types

MODULE_NAME = "num2words"


def install() -> bool:
    """Put our implementation in `sys.modules` as `num2words`. Idempotent.

    Returns True if this call installed it, False if it was already there --
    either from a previous call or because something imported the real package
    first, which is the case worth knowing about.
    """
    if MODULE_NAME in sys.modules:
        return False

    from voiceagent.text import numbers

    module = types.ModuleType(MODULE_NAME)
    module.__doc__ = (
        "Permissively-licensed stand-in for num2words, installed by "
        "voiceagent.text.num2words_shim. See voiceagent.text.numbers."
    )
    module.num2words = numbers.num2words
    # misaki only imports the callable, but these are the library's other public
    # names and something reaching for one should get a working function rather
    # than an AttributeError that looks like a broken install.
    module.CONVERTER_CLASSES = {}
    module.__voiceagent_shim__ = True
    sys.modules[MODULE_NAME] = module
    return True


def is_installed() -> bool:
    """True when the name `num2words` resolves to our shim rather than the package."""
    module = sys.modules.get(MODULE_NAME)
    return bool(module is not None and getattr(module, "__voiceagent_shim__", False))
