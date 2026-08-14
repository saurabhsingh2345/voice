"""Route text to the right TTS engine, and keep only one of them in memory.

This exists because of arithmetic, not elegance. On an 18 GiB machine the
English pipeline already sits at ~5.9 GiB; adding an Indic TTS and an Indic ASR
resident at the same time projects to ~12.9 GiB, over the budget ceiling and far
over what is actually free. Holding every engine loaded is not an option, so the
router evicts the previous one before loading the next.

The cost is a reload whenever the language alternates, which is why
`keep_resident` exists: if a machine ever has the headroom, flip it and the
eviction stops.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from voiceagent.text.detect import detect
from voiceagent.tts.base import AudioChunk, TTSEngine


@dataclass
class Route:
    """One language family and the engine that speaks it."""

    languages: frozenset[str]
    factory: Callable[[], TTSEngine]
    """Deferred construction -- nothing is instantiated until it is needed."""
    normalizer: Callable[[str], str] | None = None
    """Per-language text normalization applied before synthesis."""
    label: str = ""
    _engine: TTSEngine | None = field(default=None, repr=False)

    def speaks(self, language: str) -> bool:
        return language in self.languages


class TTSRouter:
    def __init__(self, routes: list[Route], keep_resident: bool = False) -> None:
        self.routes = routes
        self.keep_resident = keep_resident
        self._active: Route | None = None
        self.stats: dict[str, float] = {}

    # --- routing ----------------------------------------------------------

    def route_for(self, text: str) -> tuple[Route, str]:
        detection = detect(text)
        for route in self.routes:
            if route.speaks(detection.language):
                return route, detection.language
        # No Indic engine configured for this script; fall back to the first
        # route rather than failing, and let the caller see the mismatch.
        return self.routes[0], detection.language

    def _activate(self, route: Route) -> TTSEngine:
        if self._active is route and route._engine is not None:
            return route._engine

        if not self.keep_resident and self._active is not None and self._active is not route:
            # An engine that holds no memory on this machine is never worth
            # evicting; see TTSEngine.evictable. This is what makes a remote
            # engine actually stay resident on the machine running it -- without
            # it, alternating languages would tell the service to unload after
            # every English turn, reintroducing the 15-30 s reload that moving
            # the engine off this machine removed.
            #
            # getattr with a True default, not `previous.evictable`: engines are
            # duck-typed here as well as subclassed, and the default has to be
            # the safe direction. An engine that does not declare itself gets
            # evicted, which preserves the memory ceiling; defaulting the other
            # way would silently keep two models resident the first time someone
            # plugged in a backend that predates this flag.
            previous = self._active._engine
            if previous is not None and getattr(previous, "evictable", True):
                previous.unload()
                self._active._engine = None

        if route._engine is None:
            started = time.perf_counter()
            route._engine = route.factory()
            route._engine.load()
            self.stats[f"load_{route.label}_s"] = time.perf_counter() - started

        self._active = route
        return route._engine

    # --- synthesis --------------------------------------------------------

    def prepare(self, text: str) -> tuple[TTSEngine, str, str]:
        """Pick the engine, normalize the text, and return both."""
        route, language = self.route_for(text)
        engine = self._activate(route)
        prepared = route.normalizer(text) if route.normalizer else text
        return engine, prepared, language

    async def synthesize(
        self, text: str, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        engine, prepared, _ = self.prepare(text)
        async for chunk in engine.synthesize(prepared, voice=voice):
            yield chunk

    def unload(self) -> None:
        for route in self.routes:
            if route._engine is not None:
                route._engine.unload()
                route._engine = None
        self._active = None


# --- default wiring -------------------------------------------------------


def build_default_router(keep_resident: bool = False) -> TTSRouter:
    """English through Kokoro, Indic languages through an Indic-native engine.

    The Indic route is declared even when its weights are not present, so the
    routing decision is testable and the failure is a clear load error rather
    than silently speaking Hindi with an English voice.

    The route still claims all ten Indic scripts even though the engine behind
    it now speaks only Hindi. That is deliberate. Narrowing the route to `{"hi"}`
    would send Bengali to `route_for`'s fallback, which is the first route ---
    this one --- and it would be read aloud by a Hindi voice with nothing
    logged. Claiming the script and raising `UnsupportedLanguage` from the
    engine names the language in the error instead. See
    `chatterbox_indic.ChatterboxIndicEngine._require_hindi`.
    """
    from voiceagent.text.normalize_hi import normalize as normalize_hi

    def kokoro() -> TTSEngine:
        from voiceagent.tts.kokoro_engine import KokoroEngine

        return KokoroEngine()

    def indic() -> TTSEngine:
        from voiceagent.tts.chatterbox_indic import ChatterboxIndicEngine

        return ChatterboxIndicEngine()

    return TTSRouter(
        routes=[
            Route(
                languages=frozenset({"hi", "mr", "bn", "ta", "te", "kn", "ml", "gu", "pa", "or"}),
                factory=indic,
                normalizer=normalize_hi,
                label="indic",
            ),
            Route(languages=frozenset({"en"}), factory=kokoro, label="kokoro"),
        ],
        keep_resident=keep_resident,
    )
