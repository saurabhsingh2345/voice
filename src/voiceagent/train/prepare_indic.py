"""Make IndicF5 fine-tunable by f5-tts's stock trainer.

    uv run python -m voiceagent.train.prepare_indic <dataset_name>

Three things stop `f5-tts_finetune-cli` from fine-tuning IndicF5, and none of them
announce themselves as the cause of a bad model. This script fixes all three and
prints the command to run.

1. ARCHITECTURE FLAGS. `configs/F5TTS_v1_Base.yaml` sets `text_mask_padding: True`
   and `pe_attn_head: null`. IndicF5 was trained against an older f5-tts where the
   rotary embedding landed on head 0 only and the text path was unmasked, so it
   needs `pe_attn_head: 1` and `text_mask_padding: False` -- the same two flags
   `tts/indic_engine.py` pins, for the same reason. Every tensor still loads with
   the wrong flags because the shapes are unchanged; the symptom at inference was
   fluent babble in a random language (Arabic, Indonesian, Welsh). Fine-tuning
   from that state would not error either. It would spend hours destroying the
   pretrained knowledge and hand back a model that had learned to babble in your
   voice.

2. TOKENIZER. IndicF5 ships its own 2545-entry vocab. The default `pinyin`
   tokenizer builds a different vocabulary and therefore a different embedding
   size, so the text embedding would not match the checkpoint.

3. CHECKPOINT KEY LAYOUT. IndicF5 was saved through an EMA wrapper around a
   *compiled* module, so its keys read `ema_model._orig_mod.transformer...` and
   there are 447 of them. `Trainer.load_checkpoint` does a strict
   `load_state_dict` against an `EMA` whose 366 keys read `ema_model.transformer...`
   plus `initted` and `step`. That raises rather than silently mistraining, which
   is the one merciful part of this, but it does mean the stock path stops dead.

Also moves the checkpoint directory out of the virtualenv. The trainer computes it
as `files("f5_tts")/"../../ckpts/<name>"`, which resolves inside `.venv` -- so a
`uv sync` can delete a fine-tune that took hours. This symlinks it into `data/`,
which is gitignored and survives.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.resources import files
from pathlib import Path

REPO = "ai4bharat/IndicF5"

#: The two flags that matter. Kept as a dict so the config writer and any future
#: check read the same values, rather than two copies that can disagree.
INDICF5_ARCH_OVERRIDES = {"pe_attn_head": 1, "text_mask_padding": False}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CKPTS = PROJECT_ROOT / "data" / "ckpts"


def locate() -> tuple[Path, Path]:
    """Return (checkpoint, vocab), downloading them if they are not cached."""
    from huggingface_hub import hf_hub_download

    checkpoint = Path(hf_hub_download(REPO, filename="model.safetensors"))
    vocab = Path(hf_hub_download(REPO, filename="checkpoints/vocab.txt"))
    return checkpoint, vocab


def expected_ema_keys(vocab: Path) -> set[str]:
    """Build the model the trainer will build, and ask it what keys it wants.

    Derived rather than hardcoded: if f5-tts changes its module layout, this fails
    loudly at preparation time instead of producing a checkpoint that loads into
    the wrong places.
    """
    from ema_pytorch import EMA
    from f5_tts.infer import utils_infer as U
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    vocab_char_map, vocab_size = get_tokenizer(str(vocab), "custom")
    model = CFM(
        transformer=DiT(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
            text_num_embeds=vocab_size, mel_dim=U.n_mel_channels,
            **INDICF5_ARCH_OVERRIDES,
        ),
        vocab_char_map=vocab_char_map,
    )
    return set(EMA(model, include_online_model=False).state_dict().keys()), vocab_size


def convert_checkpoint(checkpoint: Path, vocab: Path, destination: Path) -> tuple[int, int]:
    """Rewrite IndicF5's checkpoint into the layout the trainer loads strictly.

    Returns (matched, dropped). Raises if any key the trainer requires is missing
    after the rewrite -- a partially-matching checkpoint is the failure mode worth
    refusing, because training would start from partly-random weights and only look
    wrong hours later.
    """
    import torch
    from safetensors.torch import load_file, save_file

    wanted, _ = expected_ema_keys(vocab)
    raw = load_file(str(checkpoint), device="cpu")

    renamed = {
        key.replace("ema_model._orig_mod.", "ema_model."): value for key, value in raw.items()
    }
    kept = {key: value for key, value in renamed.items() if key in wanted}
    dropped = len(renamed) - len(kept)

    # `initted` and `step` are EMA bookkeeping, not weights, and are absent from a
    # checkpoint saved this way. The trainer's strict load needs them present.
    if "initted" in wanted and "initted" not in kept:
        kept["initted"] = torch.tensor(True)
    if "step" in wanted and "step" not in kept:
        kept["step"] = torch.tensor(0)

    missing = wanted - set(kept)
    if missing:
        raise SystemExit(
            f"Refusing to write a partial checkpoint: {len(missing)} keys the trainer "
            f"requires are absent after the rewrite, e.g. {sorted(missing)[:3]}. "
            "Training from this would start partly from random weights and only look "
            "wrong hours later."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(destination))
    return len(kept), dropped


def write_config(dataset_name: str, vocab: Path, destination: Path) -> Path:
    """Copy the base config with the Indic architecture and tokenizer set."""
    import yaml

    source = Path(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
    config = yaml.safe_load(source.read_text())

    config["model"]["arch"].update(INDICF5_ARCH_OVERRIDES)
    config["model"]["tokenizer"] = "custom"
    config["model"]["tokenizer_path"] = str(vocab)
    config["model"]["name"] = "IndicF5_finetune"
    # Recompute activations instead of holding them. Costs time, saves memory, and
    # memory is the binding constraint on an 18 GiB laptop.
    config["model"]["arch"]["checkpoint_activations"] = True
    config["datasets"]["name"] = dataset_name
    config["ckpts"]["logger"] = None  # no wandb account required to train locally

    destination.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return destination


def link_out_of_the_venv(name: str) -> Path:
    """Redirect one of f5-tts's venv-relative directories into the project's data/.

    f5-tts resolves both its checkpoint directory and its dataset directory from
    the package location -- `files("f5_tts")/"../../ckpts"` and `.../data` -- which
    land in `.venv/lib/python3.12/`. Two problems with leaving them there: `uv sync`
    can delete a fine-tune that took hours, and neither path is covered by the
    `data/` gitignore rule that keeps voice data out of the repository.

    A symlink is the least invasive fix -- no patching of installed code, and it
    keeps working as long as re-syncing recreates the venv rather than the target.
    """
    venv_path = Path(os.path.normpath(str(files("f5_tts").joinpath(f"../../{name}"))))
    target = PROJECT_ROOT / "data" / f"f5tts_{name}"
    target.mkdir(parents=True, exist_ok=True)

    if venv_path.is_symlink():
        if venv_path.resolve() == target.resolve():
            return venv_path
        venv_path.unlink()
    elif venv_path.exists():
        if any(venv_path.iterdir()):
            raise SystemExit(
                f"{venv_path} exists and is not empty. Move or remove it, then re-run "
                "-- refusing to replace something that may hold checkpoints or datasets."
            )
        venv_path.rmdir()

    venv_path.symlink_to(target, target_is_directory=True)
    return venv_path


def place_base_vocab(vocab: Path, venv_data: Path) -> Path:
    """Put IndicF5's vocab where `prepare_csv_wavs` looks for the base model's.

    In finetune mode that script asserts on
    `data/Emilia_ZH_EN_pinyin/vocab.txt` and copies it into the new dataset, the
    intent being "reuse the vocabulary the base model was trained with". Our base
    model is IndicF5, so its 2545-entry vocab is the correct content -- the
    directory name is just where the path is hardcoded, and it does not mean the
    data is Emilia's.
    """
    destination = venv_data / "Emilia_ZH_EN_pinyin" / "vocab.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(vocab.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_name", help="the profile id you exported, e.g. e214ec611523")
    args = parser.parse_args(argv)

    checkpoint, vocab = locate()
    _, vocab_size = expected_ema_keys(vocab)
    print(f"IndicF5 checkpoint : {checkpoint}")
    print(f"vocab              : {vocab}  ({vocab_size} entries)")

    ckpt_dir = link_out_of_the_venv("ckpts") / args.dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    venv_data = link_out_of_the_venv("data")
    print(f"checkpoints        : {ckpt_dir}")
    print(f"datasets           : {venv_data}/{args.dataset_name}_custom")
    print("                     (both symlinked out of .venv so uv sync cannot delete them)")

    base_vocab = place_base_vocab(vocab, venv_data)
    print(f"base vocab placed  : {base_vocab}")

    target = ckpt_dir / "pretrained_indicf5.safetensors"
    matched, dropped = convert_checkpoint(checkpoint, vocab, target)
    print(f"rewrote            : {matched} keys kept, {dropped} dropped -> {target.name}")

    config = write_config(args.dataset_name, vocab, ckpt_dir / "IndicF5_finetune.yaml")
    print(f"config             : {config}")
    print(f"                     pe_attn_head=1, text_mask_padding=False, tokenizer=custom")

    training_dir = PROJECT_ROOT / "data" / "training" / args.dataset_name
    print("\nNext:\n")
    # The output directory is not free choice: the trainer resolves the dataset as
    # data/<dataset_name>_<tokenizer>, so with tokenizer=custom it must be exactly
    # this. Writing it anywhere else produces a dataset the trainer cannot find.
    print(f"  uv run python -m f5_tts.train.datasets.prepare_csv_wavs \\")
    print(f"      {training_dir}/metadata.csv \\")
    print(f"      {venv_data}/{args.dataset_name}_custom\n")
    # hydra's config_path is hardcoded to f5_tts/configs, so the search path has to
    # be extended with --config-dir rather than pointing at the file. And no
    # `accelerate launch`: on one device it adds nothing and swallows the traceback.
    print(f"  uv run python -m f5_tts.train.train \\")
    print(f"      --config-dir {config.parent} --config-name {config.stem}\n")
    print("Then delete the decrypted export:\n")
    print(f"  rm -rf {training_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
