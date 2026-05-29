"""Dependency auto-installer for the cover pipeline.

Checks for required packages and installs them if missing.
Handles both pip packages and model downloads.
"""

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


# Package name -> import name mapping (when they differ)
_PACKAGE_IMPORT_MAP = {
    "audio-separator": "audio_separator",
    "demucs": "demucs",
    "librosa": "librosa",
    "madmom": "madmom",
    "allin1": "allin1",
    "music-source-separation-training": "msst",
}

# Packages with their pip install specifiers
REQUIRED_PACKAGES = {
    # Stem separation
    "demucs": "demucs>=4.0.0",
    # MIR analysis
    "librosa": "librosa>=0.10.0",
    "essentia": "essentia",
    "allin1": "allin1>=1.1.0",
    # Audio I/O (needed for torchaudio soundfile backend)
    "soundfile": "soundfile>=0.13.0",
    # Professional audio effects for mixing
    "pedalboard": "pedalboard",
    # DCW quality correction during diffusion
    "pytorch_wavelets": "pytorch_wavelets",
    "PyWavelets": "PyWavelets",
    "setuptools": "setuptools<70",  # Required by pytorch_wavelets for pkg_resources
}

# Optional packages (installed on demand)
# IMPORTANT: Do NOT install onnxruntime-gpu — it conflicts with xl-sft's
# 50-step diffusion loop and produces garbled/distorted audio.
# audio-separator works fine without [gpu] extra (uses PyTorch backend).
# Install onnxruntime (CPU only) for audio-separator's import requirement.
OPTIONAL_PACKAGES = {
    "onnxruntime": "onnxruntime",
    "audio-separator": "audio-separator>=0.17.0",
    "qwen-omni-utils": "qwen-omni-utils",
    "laion-clap": "laion-clap",
}

# Special install procedures for packages that need custom build steps
_SPECIAL_INSTALLS = {
    "madmom": [
        ["uv", "pip", "install", "--python", ".venv/bin/python", "cython", "numpy"],
        ["uv", "pip", "install", "--python", ".venv/bin/python",
         "git+https://github.com/CPJKU/madmom.git@main", "--no-build-isolation"],
    ],
}

# SongFormer setup
_SONGFORMER_DIR = Path(__file__).parent / "SongFormer"
_SONGFORMER_DEPS = [
    "muq", "msaf", "omegaconf", "safetensors", "ema-pytorch",
    "x-transformers", "requests", "tqdm",
]

# Model checkpoints and their download info
MODEL_REGISTRY = {
    "mel_band_roformer": {
        "filename": "mel_band_roformer_ep_125_sdr_11.2069.ckpt",
        "url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
        "size_mb": 430,
        "description": "Mel-Band RoFormer vocal separation (SDR 11.2)",
    },
    "htdemucs_ft": {
        "filename": "htdemucs_ft",
        "url": None,  # Downloaded via demucs CLI/torch.hub
        "size_mb": 320,
        "description": "Demucs v4 fine-tuned (4-stem separation)",
    },
}


def _get_models_dir() -> Path:
    """Return the models directory for the cover pipeline."""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _is_package_installed(package_name: str) -> bool:
    """Check if a Python package is fully importable."""
    import_name = _PACKAGE_IMPORT_MAP.get(package_name, package_name)
    try:
        mod = importlib.import_module(import_name)
        # For audio-separator, verify the Separator class is importable
        if import_name == "audio_separator":
            from audio_separator.separator import Separator
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _install_package(pip_spec: str) -> bool:
    """Install a package via uv pip install into the project .venv."""
    logger.info(f"Installing: {pip_spec}")
    try:
        # Use --python to target the project .venv explicitly
        # (avoids installing into conda env when one is active)
        venv_python = Path(__file__).parent.parent.parent / ".venv" / "bin" / "python"
        cmd = ["uv", "pip", "install", pip_spec]
        if venv_python.exists():
            cmd = ["uv", "pip", "install", "--python", str(venv_python), pip_spec]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"Installed via uv: {pip_spec}")
            return True
        else:
            logger.error(f"Failed to install {pip_spec}: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.error("uv not found. Install uv first.")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout installing {pip_spec}")
        return False


def _install_special(package_name: str, commands: list[list[str]]) -> bool:
    """Install a package that requires multiple build steps."""
    logger.info(f"Installing (special): {package_name}")
    for cmd in commands:
        logger.info(f"  Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                logger.error(f"  Step failed: {result.stderr[:300]}")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"  Step error: {e}")
            return False
    logger.info(f"Installed (special): {package_name}")
    return True


def _download_model(model_key: str) -> Optional[Path]:
    """Download a model checkpoint if not already present."""
    info = MODEL_REGISTRY.get(model_key)
    if not info:
        logger.error(f"Unknown model: {model_key}")
        return None

    models_dir = _get_models_dir()
    model_path = models_dir / info["filename"]

    if model_path.exists():
        logger.info(f"Model already exists: {model_path}")
        return model_path

    if info["url"] is None:
        # Model is downloaded via its own tooling (e.g., demucs)
        logger.info(f"Model '{model_key}' is managed by its package (auto-downloads on first use)")
        return model_path

    logger.info(f"Downloading {info['description']} (~{info['size_mb']}MB)...")

    try:
        import urllib.request

        urllib.request.urlretrieve(info["url"], str(model_path))
        logger.info(f"Downloaded: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download {model_key}: {e}")
        # Try with requests as fallback
        try:
            import requests

            response = requests.get(info["url"], stream=True, timeout=600)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded (via requests): {model_path}")
            return model_path
        except Exception as e2:
            logger.error(f"Fallback download also failed: {e2}")
            return None


def _ensure_ffmpeg() -> str:
    """Check for ffmpeg and install if missing."""
    import shutil

    if shutil.which("ffmpeg"):
        logger.info("ffmpeg already installed")
        return "already_installed"

    logger.info("ffmpeg not found, installing static binary...")
    import platform

    if platform.system() != "Linux":
        logger.warning("Auto-install only supports Linux. Install ffmpeg manually.")
        return "manual_install_needed"

    try:
        arch = platform.machine()
        if arch == "x86_64":
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        elif arch == "aarch64":
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
        else:
            logger.warning(f"Unsupported architecture: {arch}")
            return "unsupported_arch"

        result = subprocess.run(
            ["bash", "-c", f"curl -L {url} | tar xJ && sudo cp ffmpeg-*-static/ffmpeg /usr/local/bin/ && sudo cp ffmpeg-*-static/ffprobe /usr/local/bin/ && rm -rf ffmpeg-*-static"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and shutil.which("ffmpeg"):
            logger.info("ffmpeg installed successfully")
            return "installed"
        else:
            logger.error(f"ffmpeg install failed: {result.stderr[:200]}")
            return "failed"
    except Exception as e:
        logger.error(f"ffmpeg install error: {e}")
        return "failed"


def _ensure_torchaudio_backend() -> str:
    """Set TORCHAUDIO_BACKEND=soundfile to avoid torchcodec/ffmpeg shared lib issues."""
    import os

    os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

    # Also redirect UV cache to SageMaker volume (root disk often too small)
    if not os.environ.get("UV_CACHE_DIR"):
        sagemaker_uv_cache = "/home/ec2-user/SageMaker/.cache/uv"
        if Path(sagemaker_uv_cache).parent.parent.exists():
            os.environ["UV_CACHE_DIR"] = sagemaker_uv_cache

    logger.info("Set TORCHAUDIO_BACKEND=soundfile")
    return "set"


def _ensure_hf_cache() -> str:
    """Ensure HuggingFace cache is on a volume with space (not root disk)."""
    import os
    import shutil

    # Always redirect to SageMaker volume if it exists (root disk is often small)
    sagemaker_cache = "/home/ec2-user/SageMaker/.cache/huggingface"
    if Path(sagemaker_cache).parent.parent.exists():
        Path(sagemaker_cache).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = sagemaker_cache
        os.environ["TRANSFORMERS_CACHE"] = sagemaker_cache
        logger.info(f"Set HF_HOME={sagemaker_cache}")
        return "set"

    # If HF_HOME is already set, respect it
    if os.environ.get("HF_HOME"):
        logger.info(f"HF_HOME already set: {os.environ['HF_HOME']}")
        return "already_set"

    return "ok"


def _ensure_songformer() -> str:
    """Clone SongFormer repo and install its dependencies if not present."""
    if _SONGFORMER_DIR.exists() and (_SONGFORMER_DIR / "src").exists():
        logger.info("SongFormer already cloned")
        # Ensure deps are installed
        for dep in _SONGFORMER_DEPS:
            if not _is_package_installed(dep):
                _install_package(dep)
        return "already_installed"

    logger.info("Cloning SongFormer repository...")
    try:
        result = subprocess.run(
            [
                "git", "clone", "--recurse-submodules",
                "https://github.com/ASLP-lab/SongFormer.git",
                str(_SONGFORMER_DIR),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"Failed to clone SongFormer: {result.stderr[:200]}")
            return "failed"

        logger.info("Installing SongFormer dependencies...")
        deps_str = " ".join(_SONGFORMER_DEPS)
        venv_python = Path(__file__).parent.parent.parent / ".venv" / "bin" / "python"
        cmd = ["uv", "pip", "install", "--python", str(venv_python)] + _SONGFORMER_DEPS
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"Some SongFormer deps failed: {result.stderr[:200]}")

        logger.info("SongFormer installed successfully")
        return "installed"

    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error(f"SongFormer setup error: {e}")
        return "failed"


def _ensure_acestep_models() -> str:
    """Download required ACE-Step models if not present."""
    checkpoints_dir = Path(__file__).parent.parent.parent / "checkpoints"

    required_models = [
        "acestep-v15-xl-sft",
        "acestep-5Hz-lm-4B",
    ]

    missing = []
    for model in required_models:
        model_path = checkpoints_dir / model
        if not model_path.exists():
            missing.append(model)

    if not missing:
        logger.info(f"ACE-Step models present: {required_models}")
        return "already_installed"

    logger.info(f"Downloading ACE-Step models: {missing}")
    for model in missing:
        try:
            result = subprocess.run(
                ["uv", "run", "acestep-download", "--model", model],
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=str(Path(__file__).parent.parent.parent),
            )
            if result.returncode == 0:
                logger.info(f"Downloaded: {model}")
            else:
                logger.error(f"Failed to download {model}: {result.stderr[:200]}")
                return "failed"
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"Download error for {model}: {e}")
            return "failed"

    return "installed"


def ensure_dependencies(include_optional: bool = False) -> dict:
    """Install all required dependencies and download models.

    Returns:
        Dictionary with installation status for each component.
    """
    status = {"packages": {}, "models": {}, "system": {}}

    # System dependencies (ffmpeg, torchaudio backend, HF cache location)
    status["system"]["ffmpeg"] = _ensure_ffmpeg()
    status["system"]["torchaudio_backend"] = _ensure_torchaudio_backend()
    status["system"]["hf_cache"] = _ensure_hf_cache()
    status["system"]["acestep_models"] = _ensure_acestep_models()
    status["system"]["songformer"] = _ensure_songformer()

    # Install required packages
    packages_to_install = dict(REQUIRED_PACKAGES)
    if include_optional:
        packages_to_install.update(OPTIONAL_PACKAGES)

    for package_name, pip_spec in packages_to_install.items():
        if _is_package_installed(package_name):
            status["packages"][package_name] = "already_installed"
        else:
            success = _install_package(pip_spec)
            status["packages"][package_name] = "installed" if success else "failed"

    # Install packages with special build procedures
    for package_name, commands in _SPECIAL_INSTALLS.items():
        if _is_package_installed(package_name):
            status["packages"][package_name] = "already_installed"
        else:
            success = _install_special(package_name, commands)
            status["packages"][package_name] = "installed" if success else "failed"

    # Download models
    for model_key in MODEL_REGISTRY:
        path = _download_model(model_key)
        status["models"][model_key] = str(path) if path else "failed"

    # Summary
    failed = [k for k, v in status["packages"].items() if v == "failed"]
    failed += [k for k, v in status["models"].items() if v == "failed"]

    if failed:
        logger.warning(f"Some components failed to install: {failed}")
    else:
        logger.info("All dependencies and models ready.")

    return status


def get_model_path(model_key: str) -> Optional[Path]:
    """Get the path to a downloaded model, downloading if needed."""
    models_dir = _get_models_dir()
    info = MODEL_REGISTRY.get(model_key)
    if not info:
        return None

    model_path = models_dir / info["filename"]
    if model_path.exists():
        return model_path

    return _download_model(model_key)
