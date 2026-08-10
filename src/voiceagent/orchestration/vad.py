"""Silero VAD: decide when the user starts and stops speaking.

Turn-taking is the part of a voice agent people notice most. Cut in too early
and you talk over them mid-sentence; wait too long and the agent feels slow. So
the two thresholds here are deliberately asymmetric:

  * Speech *onset* is reported almost immediately -- it is what triggers
    barge-in, and cancelling playback a beat late is very audible.
  * Speech *end* requires a run of silence, because natural speech is full of
    short gaps and treating the first one as end-of-turn produces the classic
    "interrupts you when you pause to think" failure.

Runs the ONNX model so the hot path does not go through torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

#: Silero requires exactly 512-sample frames at 16 kHz (32 ms).
FRAME_SAMPLES = 512
SAMPLE_RATE = 16_000
FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000


class VADEvent(str, Enum):
    NONE = "none"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VADConfig:
    threshold: float = 0.5
    """Speech probability above which a frame counts as speech."""
    min_speech_ms: float = 96.0
    """Speech must persist this long before we call it a turn (3 frames)."""
    min_silence_ms: float = 640.0
    """Silence needed to end a turn. Shorter than this and normal pauses cut you off."""
    speech_pad_ms: float = 192.0
    """Audio kept before onset, so the first phoneme is not clipped from the STT."""


class SileroVAD:
    """Frame-by-frame voice activity detection with hysteresis."""

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self._model = None
        self._torch = None
        self._triggered = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._pad: list[np.ndarray] = []
        self._residual = np.zeros(0, dtype=np.float32)

    def load(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        # Inference runs through ONNX Runtime, but silero's wrapper validates
        # its input as a torch tensor either way. from_numpy is zero-copy, so
        # this costs nothing per frame.
        self._torch = torch
        self._model = load_silero_vad(onnx=True)

    def reset(self) -> None:
        """Clear state between turns without reloading the model."""
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()
        self._triggered = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._pad.clear()
        self._residual = np.zeros(0, dtype=np.float32)

    @property
    def is_speaking(self) -> bool:
        return self._triggered

    @property
    def _min_speech_frames(self) -> int:
        return max(1, int(self.config.min_speech_ms / FRAME_MS))

    @property
    def _min_silence_frames(self) -> int:
        return max(1, int(self.config.min_silence_ms / FRAME_MS))

    @property
    def _max_pad_frames(self) -> int:
        return max(1, int(self.config.speech_pad_ms / FRAME_MS))

    def process(self, audio: np.ndarray) -> list[tuple[VADEvent, np.ndarray]]:
        """Feed arbitrary-length audio; get back events with their audio.

        Callers hand us whatever block size the sound card produces, which is
        rarely a multiple of 512, so leftover samples are carried across calls.
        """
        if self._model is None:
            raise RuntimeError("load() must be called before process()")

        buffer = np.concatenate([self._residual, audio.astype(np.float32)])
        events: list[tuple[VADEvent, np.ndarray]] = []
        offset = 0

        while offset + FRAME_SAMPLES <= len(buffer):
            frame = buffer[offset : offset + FRAME_SAMPLES]
            offset += FRAME_SAMPLES
            events.extend(self._process_frame(frame))

        self._residual = buffer[offset:]
        return events

    def _process_frame(self, frame: np.ndarray) -> list[tuple[VADEvent, np.ndarray]]:
        tensor = self._torch.from_numpy(frame).unsqueeze(0)
        probability = float(self._model(tensor, SAMPLE_RATE).item())
        is_speech = probability >= self.config.threshold
        events = []

        if not self._triggered:
            # Keep a rolling pre-roll so the STT sees the start of the word.
            self._pad.append(frame)
            if len(self._pad) > self._max_pad_frames:
                self._pad.pop(0)

            if is_speech:
                self._speech_frames += 1
                if self._speech_frames >= self._min_speech_frames:
                    self._triggered = True
                    self._silence_frames = 0
                    preroll = np.concatenate(self._pad) if self._pad else frame
                    self._pad.clear()
                    events.append((VADEvent.SPEECH_START, preroll))
            else:
                self._speech_frames = 0
        else:
            events.append((VADEvent.NONE, frame))
            if is_speech:
                self._silence_frames = 0
            else:
                self._silence_frames += 1
                if self._silence_frames >= self._min_silence_frames:
                    self._triggered = False
                    self._speech_frames = 0
                    events.append((VADEvent.SPEECH_END, np.zeros(0, dtype=np.float32)))

        return events
