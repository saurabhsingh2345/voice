"""Recording prompts for voice enrolment. See prompts.py.

This package used to hold `prepare_indic.py`, which made IndicF5 fine-tunable by
f5-tts's stock trainer. Both went when the Hindi path moved to Chatterbox
Multilingual: fine-tuning was f5-tts-specific and Chatterbox clones zero-shot
from the reference clip, so there is no per-voice checkpoint to train.

`prompts.py` outlived it because the prompts are about the *speaker*, not the
model --- they are the script someone reads to enrol a voice, and a better
reference clip still improves the clone.
"""
