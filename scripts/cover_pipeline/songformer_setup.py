"""SongFormer environment setup, checkpoint download, and model loading.

Handles sys.path configuration for the vendored SongFormer repo,
downloads pretrained checkpoints on first use, and provides
load/unload functions for the three-model stack.
"""

import gc
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from loguru import logger

# Paths relative to this file — SongFormer repo cloned into cover_pipeline/SongFormer/
_THIS_DIR = Path(__file__).resolve().parent
_SONGFORMER_SRC = _THIS_DIR / "SongFormer" / "src" / "SongFormer"
_THIRD_PARTY = _THIS_DIR / "SongFormer" / "src" / "third_party"
_CKPTS_DIR = _SONGFORMER_SRC / "ckpts"

MUSICFM_DIR = _CKPTS_DIR / "MusicFM"
SONGFORMER_CKPT = _CKPTS_DIR / "SongFormer.safetensors"
CONFIG_PATH = "SongFormer.yaml"
MODEL_NAME = "SongFormer"


class SongFormerStack(NamedTuple):
    """Container for the three loaded models plus config."""

    muq: object
    musicfm: object
    model: object
    hp: object


def setup_paths() -> None:
    """Add SongFormer source and third-party dirs to sys.path."""
    _musicfm_dir = _THIRD_PARTY / "musicfm"
    if _musicfm_dir.exists() and not any(_musicfm_dir.iterdir()):
        logger.info("SongFormer submodules not initialized, running git submodule update")
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=str(_THIS_DIR / "SongFormer"),
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Failed to initialize submodules: {e}")

    paths_to_add = [str(_THIRD_PARTY), str(_SONGFORMER_SRC)]
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)
            logger.debug(f"Added to sys.path: {p}")


def ensure_checkpoints() -> None:
    """Download MusicFM and SongFormer checkpoints if missing."""
    files = [
        (
            "https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json",
            MUSICFM_DIR / "msd_stats.json",
        ),
        (
            "https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt",
            MUSICFM_DIR / "pretrained_msd.pt",
        ),
        (
            "https://huggingface.co/ASLP-lab/SongFormer/resolve/main/SongFormer.safetensors",
            SONGFORMER_CKPT,
        ),
    ]
    all_present = all(f[1].exists() for f in files)
    if all_present:
        logger.info("SongFormer checkpoints already present")
        return

    import requests
    from tqdm import tqdm

    for url, dest in files:
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {dest.name} ...")
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, desc=dest.name,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    logger.info("SongFormer checkpoints ready")


def load_models(device: str = "cuda:0") -> SongFormerStack:
    """Load MuQ, MusicFM, and SongFormer onto the given device."""
    setup_paths()
    ensure_checkpoints()

    import importlib

    import torch
    from ema_pytorch import EMA
    from muq import MuQ
    from musicfm.model.musicfm_25hz import MusicFM25Hz
    from omegaconf import OmegaConf

    logger.info("Loading MuQ ...")
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq = muq.to(device).eval()

    logger.info("Loading MusicFM ...")
    musicfm = MusicFM25Hz(
        is_flash=False,
        stat_path=str(MUSICFM_DIR / "msd_stats.json"),
        model_path=str(MUSICFM_DIR / "pretrained_msd.pt"),
    )
    musicfm = musicfm.to(device).eval()

    logger.info("Loading SongFormer transformer ...")
    hp = OmegaConf.load(str(_SONGFORMER_SRC / "configs" / CONFIG_PATH))
    import numpy as np
    import scipy
    if not hasattr(scipy, "inf"):
        scipy.inf = np.inf
    module = importlib.import_module("models." + MODEL_NAME)
    Model = getattr(module, "Model")
    model = Model(hp)

    from safetensors.torch import load_file
    ckpt = {"model_ema": load_file(str(SONGFORMER_CKPT), device=device)}
    model_ema = EMA(model, include_online_model=False)
    model_ema.load_state_dict(ckpt["model_ema"])
    model.load_state_dict(model_ema.ema_model.state_dict())
    model.to(device).eval()

    logger.info("SongFormer stack loaded")
    return SongFormerStack(muq=muq, musicfm=musicfm, model=model, hp=hp)


def unload_models(stack: SongFormerStack) -> None:
    """Free GPU memory for all three models."""
    import torch

    del stack
    gc.collect()
    for key in list(sys.modules):
        if key.startswith(("muq", "musicfm", "ema_pytorch")):
            del sys.modules[key]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("SongFormer stack unloaded, GPU memory freed")
