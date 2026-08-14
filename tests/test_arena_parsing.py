"""The two places SpeechArenaBench can be misread without anything erroring.

Both functions under test turn a released string into a label the verdict is
computed from, and both fail silently when wrong: a mis-parsed vote becomes a
tie, a mis-classified sentence moves between subsets, and either way the output
is a full ranking that looks fine.

The vote strings here are copied from the real Hindi shards, not invented. The
released `preference_model` column does not hold "Model A"/"Model B" the way the
dataset card describes -- it holds model names, and encodes "both good" by
listing both of them. Anything written against the card rather than the data
counts every "Bulbul V3 Beta, Speech 2.8 HD" as unparseable.

No network: these are pure functions over strings.
"""

from __future__ import annotations

import pytest

from voiceagent.eval.arena import classify_subset, parse_preference

BULBUL = "Bulbul V3 Beta"
SPEECH = "Speech 2.8 HD"
GEMINI = "Gemini 2.5 Pro TTS"


class TestParsePreference:
    def test_a_single_name_is_a_win_for_that_side(self):
        assert parse_preference(SPEECH, BULBUL, SPEECH) == "b"
        assert parse_preference(BULBUL, BULBUL, SPEECH) == "a"

    def test_both_names_listed_is_both_good(self):
        """The real encoding, and the one a card-driven parser drops."""
        assert parse_preference(f"{BULBUL}, {SPEECH}", BULBUL, SPEECH) == "both_good"
        # Order in the string does not track model_a/model_b.
        assert parse_preference(f"{SPEECH}, {BULBUL}", BULBUL, SPEECH) == "both_good"

    def test_tie_string_is_both_bad(self):
        assert parse_preference("Tie / No Preference", BULBUL, SPEECH) == "both_bad"

    def test_a_name_that_is_a_substring_of_the_other_is_not_confused(self):
        """Guards the obvious `in` bug.

        Naming systems by substring match would let "Sonic 3" hit inside a
        hypothetical "Sonic 3 Turbo" and silently award the wrong side. The
        parser splits on commas and compares whole names, and this pins that.
        """
        assert parse_preference("Sonic 3 Turbo", "Sonic 3", "Sonic 3 Turbo") == "b"
        assert parse_preference("Sonic 3", "Sonic 3", "Sonic 3 Turbo") == "a"

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_missing_preference_is_not_a_tie(self, value):
        """A row with no verdict is missing data, and counting it as a tie would
        quietly drag every system towards the middle."""
        assert parse_preference(value, BULBUL, SPEECH) is None

    def test_a_name_from_neither_side_is_rejected(self):
        assert parse_preference(GEMINI, BULBUL, SPEECH) is None


class TestClassifySubset:
    def test_pure_devanagari_is_normalized(self):
        assert classify_subset("टूटे ताले तले ठंडे दूध की थाली ठहरी।") == "normalized"

    def test_mixed_script_is_codemixed(self):
        sentence = "पापा ने कहा कि नया laptop दिला देंगे, but on one condition"
        assert classify_subset(sentence) == "codemixed"

    def test_fully_romanised_hinglish_is_codemixed(self):
        """Contains no Devanagari at all, and is still code-mixed Hindi.

        This is the case that makes "does it contain Latin" the right first
        test. A classifier keyed on "Devanagari plus Latin" calls this one
        normalized and puts romanised Hinglish in the wrong bucket.
        """
        assert classify_subset("Organic synthesis lab mein, students aksar") == "codemixed"

    def test_raw_numerals_are_symbolic(self):
        assert classify_subset("इस साल 1299 लोग आए।") == "symbolic"

    def test_verbalised_numerals_are_normalized(self):
        assert classify_subset("इस साल एक हज़ार दो सौ निन्यानवे लोग आए।") == "normalized"

    def test_devanagari_digits_count_as_symbolic(self):
        assert classify_subset("इस साल १२९९ लोग आए।") == "symbolic"

    def test_latin_wins_over_symbols(self):
        """A documented ordering choice, pinned so it cannot drift silently.

        The README's code-mixed definition includes mixed-script sentences, and
        a STEM sentence written in romanised Hindi is both code-mixed and
        symbolic. It is counted code-mixed.
        """
        assert classify_subset("Organic synthesis mein CH₃COOH use hota hai") == "codemixed"
