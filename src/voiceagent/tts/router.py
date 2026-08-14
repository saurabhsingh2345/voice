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

    async def synthesize_stream(
        self, text_chunks: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Route a live token stream, deciding the language on the first sentence.

        The live loop used to hold a `KokoroEngine` directly, which is why it was
        English-only: Devanagari sent to Kokoro produces nothing usable. Routing
        here instead makes the loop bilingual on output.

        Three things this has to get right that `synthesize` does not.

        **When to decide.** Script detection is reliable but needs text, and the
        first token of a Hindi reply can easily be an English loanword. Deciding
        on the first *sentence* rather than the first token costs nothing --- the
        engine could not have started before a sentence boundary anyway --- and
        removes a whole class of "answered in the wrong voice". If the reply is
        shorter than one sentence, the flush decides.

        **Normalizing per sentence, not per token.** `normalize_hi` transliterates
        Latin to Devanagari and rewrites digits as words; both need whole words,
        and the Hindi engine measurably wants it (h1 scores 94 % raw against 98 %
        transliterated). Applying it to a token at a time would corrupt exactly
        the multi-character sequences it exists to handle.

        **Not re-deciding mid-reply.** The engine is chosen once. A reply that
        drifts from Hindi to English would otherwise evict and reload a model
        mid-sentence, which is seconds of silence; and the router holds one engine
        at a time, so it genuinely cannot do both. The first sentence wins.
        """
        from voiceagent.tts.chunker import SentenceChunker

        chunker = SentenceChunker()
        pending: list[str] = []
        engine: TTSEngine | None = None
        route: Route | None = None

        async def sentences() -> AsyncIterator[str]:
            for sentence in pending:
                yield route.normalizer(sentence) if route.normalizer else sentence
            async for text in text_chunks:
                for sentence in chunker.feed(text):
                    yield route.normalizer(sentence) if route.normalizer else sentence
            for sentence in chunker.flush():
                yield route.normalizer(sentence) if route.normalizer else sentence

        # Pull until there is a whole sentence to decide on, or the stream ends.
        async for text in text_chunks:
            pending.extend(chunker.feed(text))
            if pending:
                break
        else:
            pending.extend(chunker.flush())

        if not pending:
            return

        route, _ = self.route_for(" ".join(pending))
        engine = self._activate(route)
        async for chunk in engine.synthesize_stream(sentences(), voice=voice):
            yield chunk

    def cancel(self) -> None:
        """Stop synthesis immediately, on whichever engine is speaking.

        Required for barge-in: the live loop calls this the moment the VAD hears
        the user start talking, and it used to be calling straight through to a
        `KokoroEngine`. Only the active route can be mid-utterance, but every
        built engine is cancelled anyway --- an engine left with `_cancelled`
        False after an interrupted turn would run one more span on its next call.
        """
        for route in self.routes:
            if route._engine is not None:
                route._engine.cancel()

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
        """Chatterbox Multilingual, pointed at an enrolled voice.

        Unlike Kokoro this is a *cloning* model with no built-in speaker, so it
        cannot say anything until it has a reference clip. That is the second
        reason the live loop was English-only, and the one that survives now that
        Hindi runs at RTF 0.63: speed was never the whole story.

        The clip comes from the consent-gated store, so enrolling a voice in
        `voice-web` is what turns Hindi on. With no enrolled voice the engine
        raises its own error naming the fix; nothing is guessed and no default
        speaker is shipped, because a shipped default would be a real person's
        voice with no consent record attached to it.
        """
        from voiceagent.tts.chatterbox_indic import ChatterboxIndicEngine

        engine = ChatterboxIndicEngine()
        try:
            import io

            import soundfile as sf

            from voiceagent.voice_clone.store import VoiceProfileStore

            store = VoiceProfileStore()
            profiles = store.list()
            if profiles:
                profile = profiles[0]
                audio, rate = sf.read(
                    io.BytesIO(store.reference_audio(profile.profile_id)), dtype="float32"
                )
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                engine.set_reference(audio, profile.reference_text or "", rate)
        except Exception:  # noqa: BLE001
            # No store, no profile, or an unreadable clip. Left unset so the
            # engine's own message explains it rather than this failing at
            # construction, which would take the English route down with it.
            pass
        return engine

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
