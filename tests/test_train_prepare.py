"""The checkpoint rewrite that makes IndicF5 fine-tunable.

This is the part worth testing without the 1.4 GB checkpoint present. If the key
transform drifts, `Trainer.load_checkpoint` does a strict `load_state_dict`, so the
failure is loud -- but the *partial* case is not: a rewrite that matched most keys
would train from partly-random weights and only look wrong hours later. So the
refusal is asserted, not just the success.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file

from pathlib import Path

from voiceagent.train import prepare_indic


def fake_indicf5_checkpoint(path, keys, extra=()):
    """A checkpoint shaped like IndicF5's: EMA around a compiled module."""
    state = {f"ema_model._orig_mod.{k}": torch.zeros(1) for k in keys}
    state.update({k: torch.zeros(1) for k in extra})
    save_file(state, str(path))
    return path


@pytest.fixture
def wanted(monkeypatch):
    """Stand in for the 366 keys the trainer's EMA wrapper asks for."""
    keys = {"initted", "step"} | {f"ema_model.transformer.layer{i}.weight" for i in range(8)}
    monkeypatch.setattr(prepare_indic, "expected_ema_keys", lambda vocab: (keys, 2545))
    return keys


def test_the_orig_mod_prefix_is_stripped(tmp_path, wanted):
    """IndicF5 keys read ema_model._orig_mod.X because it was saved through an EMA
    wrapper around a torch.compile'd module. The trainer wants ema_model.X."""
    src = fake_indicf5_checkpoint(
        tmp_path / "in.safetensors", [f"transformer.layer{i}.weight" for i in range(8)]
    )
    out = tmp_path / "out.safetensors"
    matched, dropped = prepare_indic.convert_checkpoint(src, tmp_path / "vocab.txt", out)

    assert matched == len(wanted)
    written = set(load_file(str(out)).keys())
    assert written == wanted
    assert not any("_orig_mod" in k for k in written)


def test_extra_keys_are_dropped(tmp_path, wanted):
    """The real checkpoint carries 447 keys against the 366 the trainer wants --
    mel_spec buffers and the online model. Extras must not reach a strict load."""
    src = fake_indicf5_checkpoint(
        tmp_path / "in.safetensors",
        [f"transformer.layer{i}.weight" for i in range(8)],
        extra=["ema_model._orig_mod.mel_spec.mel_stft.window", "online_model.transformer.x"],
    )
    _, dropped = prepare_indic.convert_checkpoint(src, tmp_path / "vocab.txt", tmp_path / "o.safetensors")
    assert dropped == 2


def test_ema_bookkeeping_is_synthesised(tmp_path, wanted):
    """initted and step are EMA state, not weights, and are absent from a
    checkpoint saved this way -- but a strict load still requires them."""
    src = fake_indicf5_checkpoint(
        tmp_path / "in.safetensors", [f"transformer.layer{i}.weight" for i in range(8)]
    )
    out = tmp_path / "out.safetensors"
    prepare_indic.convert_checkpoint(src, tmp_path / "vocab.txt", out)
    written = load_file(str(out))
    assert bool(written["initted"].item()) is True
    assert int(written["step"].item()) == 0


def test_a_partial_match_is_refused(tmp_path, wanted):
    """The dangerous case. Half a checkpoint loads into a model that then trains
    from partly-random weights, and nothing says so until the output is bad."""
    src = fake_indicf5_checkpoint(
        tmp_path / "in.safetensors", [f"transformer.layer{i}.weight" for i in range(3)]
    )
    with pytest.raises(SystemExit, match="partial checkpoint"):
        prepare_indic.convert_checkpoint(src, tmp_path / "vocab.txt", tmp_path / "o.safetensors")


def test_nothing_is_written_when_the_rewrite_is_refused(tmp_path, wanted):
    src = fake_indicf5_checkpoint(
        tmp_path / "in.safetensors", [f"transformer.layer{i}.weight" for i in range(3)]
    )
    out = tmp_path / "out.safetensors"
    with pytest.raises(SystemExit):
        prepare_indic.convert_checkpoint(src, tmp_path / "vocab.txt", out)
    assert not out.exists(), "a refused rewrite must not leave a broken checkpoint behind"


# --- the architecture flags ------------------------------------------------


def test_the_indic_arch_overrides_are_the_ones_the_engine_pins():
    """These two are the whole reason a prepared config is needed. With the stock
    values IndicF5 emits fluent babble in a random language, and every tensor still
    loads because only where the positional signal lands changes -- so a fine-tune
    would not error, it would spend hours destroying the pretrained weights."""
    from voiceagent.tts import indic_engine

    assert prepare_indic.INDICF5_ARCH_OVERRIDES == {"pe_attn_head": 1, "text_mask_padding": False}
    # The engine documents the same pair; keep the two in step.
    assert "pe_attn_head=1" in indic_engine.OLD_SEMANTICS
    assert "text_mask_padding=False" in indic_engine.OLD_SEMANTICS


def test_the_written_config_carries_the_overrides(tmp_path, monkeypatch):
    import yaml

    written = prepare_indic.write_config("myvoice", tmp_path / "vocab.txt", tmp_path / "cfg.yaml")
    config = yaml.safe_load(written.read_text())

    assert config["model"]["arch"]["pe_attn_head"] == 1
    assert config["model"]["arch"]["text_mask_padding"] is False
    assert config["model"]["tokenizer"] == "custom"
    assert config["model"]["tokenizer_path"] == str(tmp_path / "vocab.txt")
    assert config["datasets"]["name"] == "myvoice"
    # Recompute activations rather than hold them: memory is the binding constraint.
    assert config["model"]["arch"]["checkpoint_activations"] is True
    # wandb must not be required to train locally.
    assert config["ckpts"]["logger"] is None


# --- the checkpoint has to be where the trainer looks --------------------


def test_save_dir_is_pinned_to_where_the_checkpoint_is_written(tmp_path):
    """The mistake that cost a 22-epoch run. f5-tts has two conventions:
    f5-tts_finetune-cli uses ckpts/<dataset_name>, while f5_tts.train.train uses
    ckpts/<cfg.ckpts.save_dir> whose stock template expands to
    ckpts/<model_name>_<mel>_<tokenizer>_<dataset>. Trainer.load_checkpoint finds
    nothing in the second, does not warn, and trains from random weights -- and the
    loss curve falls just as convincingly as a real fine-tune."""
    import yaml

    written = prepare_indic.write_config("myvoice", tmp_path / "vocab.txt", tmp_path / "c.yaml")
    config = yaml.safe_load(written.read_text())
    assert config["ckpts"]["save_dir"] == "ckpts/myvoice"
    assert "${" not in config["ckpts"]["save_dir"], "an unexpanded template moves the directory"


def test_preparation_refuses_when_the_trainer_would_look_elsewhere(tmp_path, monkeypatch):
    import yaml

    config_path = tmp_path / "c.yaml"
    config_path.write_text(yaml.safe_dump({"ckpts": {"save_dir": "ckpts/somewhere-else"}}))
    checkpoint = tmp_path / "ckpts" / "myvoice" / "pretrained_indicf5.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")

    with pytest.raises(SystemExit, match="find nothing"):
        prepare_indic.verify_trainer_will_load(config_path, checkpoint)


def test_preparation_refuses_when_the_checkpoint_is_missing(tmp_path):
    import os
    import yaml
    from importlib.resources import files

    ckpts = Path(os.path.normpath(str(files("f5_tts").joinpath("../../ckpts")))) / "gone"
    config_path = tmp_path / "c.yaml"
    config_path.write_text(yaml.safe_dump({"ckpts": {"save_dir": "ckpts/gone"}}))
    with pytest.raises(SystemExit, match="No pretrained checkpoint"):
        prepare_indic.verify_trainer_will_load(config_path, ckpts / "pretrained_indicf5.safetensors")
