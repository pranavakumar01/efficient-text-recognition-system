"""Model registry: discovery, lazy loading and metadata.

The API needs to answer three questions without ever loading a model:

* which models exist on this machine?
* were they trained by this project, or are they downloaded third-party
  weights?
* what did they score during training?

Discovery therefore reads only the cheap ``history.json`` written beside each
checkpoint; the ``.pt`` file is opened on first use.  On a 4 GB GPU that is
not just an optimisation -- loading every model at start-up would not fit.

Provenance
----------
Every entry carries ``provenance``:

``trained_in_project``
    Trained by one of this project's scripts.  Its metrics are results of
    this work.
``pretrained_third_party``
    Downloaded weights (TrOCR).  A useful baseline, but its numbers belong to
    its authors, not to this project, and the UI labels it as such.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from ml.config import Config
from ml.inference.predictor import BasePredictor, CRNNPredictor, TrOCRPredictor
from ml.models.trocr_wrapper import TrOCRRecognizer, TrOCRUnavailable
from ml.utils.paths import relative_to_root, resolve_path

__all__ = ["ModelEntry", "ModelRegistry", "ModelNotFound", "ModelLoadError"]

logger = logging.getLogger(__name__)


class ModelNotFound(KeyError):
    """Requested model id is not registered."""


class ModelLoadError(RuntimeError):
    """A registered model exists but could not be loaded."""


#: Friendly labels and descriptions for the runs this project produces.
_KNOWN_RUNS: dict[str, dict[str, str]] = {
    "baseline_scratch": {
        "trained_on": "synthetic",
        "name": "CRNN+Attention (from scratch)",
        "description": (
            "CNN + BiLSTM + self-attention + CTC, trained from random initialisation on "
            "the reduced labeled subset. The control arm of the self-supervised experiment."
        ),
        "arm": "control",
    },
    "simclr_finetune": {
        "trained_on": "synthetic",
        "name": "CRNN+Attention (SimCLR pretrained)",
        "description": (
            "The same architecture and the same labeled subset, but with the CNN encoder "
            "initialised from SimCLR pretraining on unlabeled images. The treatment arm."
        ),
        "arm": "treatment",
    },
    "crnn_full": {
        "trained_on": "synthetic",
        "name": "CRNN+Attention (full labeled data)",
        "description": (
            "The same architecture trained on the complete labeled training set. Included "
            "as an upper reference point, not as part of the controlled comparison."
        ),
        "arm": "reference",
    },
    # The same experiment repeated on real photographed handwriting.
    "hw_baseline_scratch": {
        "trained_on": "handwriting",
        "name": "Handwriting CRNN+Attention (from scratch)",
        "description": (
            "Control arm on real handwriting: CNN + BiLSTM + self-attention + CTC trained "
            "from random initialisation on 2,000 labeled handwritten samples."
        ),
        "arm": "control",
    },
    "hw_simclr_finetune": {
        "trained_on": "handwriting",
        "name": "Handwriting CRNN+Attention (SimCLR pretrained)",
        "description": (
            "Treatment arm on real handwriting: the same architecture and the same 2,000 "
            "labeled samples, with the encoder initialised from SimCLR pretraining on "
            "~30,000 unlabeled handwritten images."
        ),
        "arm": "treatment",
    },
}


@dataclass(slots=True)
class ModelEntry:
    """A registered model, loaded or not."""

    id: str
    name: str
    family: str
    provenance: str
    description: str = ""
    checkpoint: Path | None = None
    available: bool = True
    unavailable_reason: str | None = None
    training_summary: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "provenance": self.provenance,
            "description": self.description,
            "checkpoint": relative_to_root(self.checkpoint) if self.checkpoint else None,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "training_summary": self.training_summary,
            **self.extra,
        }


def _read_history(directory: Path) -> dict[str, Any]:
    """Read ``history.json`` from a run directory; empty dict when absent."""
    path = directory / "history.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}


def _summarise(history: dict[str, Any]) -> dict[str, Any]:
    """Condense a training history into the few fields the UI shows."""
    if not history:
        return {}

    epochs = history.get("epochs") or []
    best_epoch_number = history.get("best_epoch")
    best_epoch: dict[str, Any] = {}
    for record in epochs:
        if record.get("epoch") == best_epoch_number:
            best_epoch = record
            break
    if not best_epoch and epochs:
        best_epoch = epochs[-1]

    notes = history.get("notes") or {}
    return {
        "epochs_completed": history.get("epochs_completed", len(epochs)),
        "best_epoch": best_epoch_number,
        "monitor": history.get("monitor"),
        "best_value": history.get("best_value"),
        "val_cer": best_epoch.get("val_cer"),
        "val_wer": best_epoch.get("val_wer"),
        "val_accuracy": best_epoch.get("val_accuracy"),
        "train_loss": best_epoch.get("train_loss"),
        "training_seconds": history.get("total_duration_s"),
        "train_samples": notes.get("train_samples"),
        "device": history.get("device"),
        "parameters": (notes.get("parameters") or {}).get("total"),
        "encoder_transfer": notes.get("encoder_transfer"),
    }


class ModelRegistry:
    """Discovers and lazily loads every available recogniser.

    Args:
        config: the loaded default config (for TrOCR ids and decoding).
        device: device the models are placed on.
        checkpoint_root: directory scanned for training runs.

    >>> registry = ModelRegistry(config, device)
    >>> [entry.id for entry in registry.entries()]
    ['simclr_finetune', 'baseline_scratch', 'trocr_printed', 'trocr_handwritten']
    >>> predictor = registry.get("baseline_scratch")   # loads now
    """

    def __init__(
        self,
        config: Config,
        device: torch.device,
        *,
        checkpoint_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.checkpoint_root = resolve_path(
            checkpoint_root or config.get("train.checkpoint_dir", "artifacts/checkpoints")
        ) or Path("artifacts/checkpoints")

        self._entries: dict[str, ModelEntry] = {}
        self._predictors: dict[str, BasePredictor] = {}
        self._lock = threading.Lock()

        self.refresh()

    # -- discovery ----------------------------------------------------------
    def refresh(self) -> None:
        """Re-scan the checkpoint directory.

        Called at start-up and after a training job finishes, so a newly
        trained model appears without restarting the server.
        """
        with self._lock:
            discovered: dict[str, ModelEntry] = {}
            for entry in self._discover_checkpoints():
                discovered[entry.id] = entry
            for entry in self._discover_trocr():
                discovered[entry.id] = entry

            # Drop cached predictors whose checkpoint has disappeared.
            for model_id in list(self._predictors):
                if model_id not in discovered:
                    logger.info("Model %r is gone; unloading it", model_id)
                    self._predictors.pop(model_id, None)

            self._entries = discovered

        logger.info(
            "Model registry: %d model(s) -- %s",
            len(self._entries),
            ", ".join(sorted(self._entries)) or "none",
        )

    def _discover_checkpoints(self) -> list[ModelEntry]:
        """Find recogniser checkpoints under the checkpoint root."""
        if not self.checkpoint_root.is_dir():
            logger.info(
                "No checkpoint directory at %s yet; only pretrained models will be offered.",
                relative_to_root(self.checkpoint_root),
            )
            return []

        entries: list[ModelEntry] = []
        for directory in sorted(self.checkpoint_root.iterdir()):
            if not directory.is_dir():
                continue

            checkpoint = directory / "best.pt"
            if not checkpoint.is_file():
                checkpoint = directory / "last.pt"
                if not checkpoint.is_file():
                    continue

            # A SimCLR pretraining run writes `encoder_best.pt` and contains no
            # recognition head, so it is not something that can transcribe.
            if (directory / "encoder_best.pt").is_file():
                logger.debug("Skipping %s: it is a SimCLR pretraining run, not a recogniser", directory.name)
                continue

            run_id = directory.name
            known = _KNOWN_RUNS.get(run_id, {})
            history = _read_history(directory)

            entries.append(
                ModelEntry(
                    id=run_id,
                    name=known.get("name", run_id.replace("_", " ").title()),
                    family="crnn_attention",
                    provenance="trained_in_project",
                    description=known.get(
                        "description", f"CRNN+Attention recogniser trained by this project ({run_id})."
                    ),
                    checkpoint=checkpoint,
                    available=True,
                    training_summary=_summarise(history),
                    extra={
                        "experiment_arm": known.get("arm", "other"),
                        "trained_on": known.get("trained_on", "unknown"),
                        "run_directory": relative_to_root(directory),
                        "has_history": bool(history),
                    },
                )
            )
        return entries

    def _discover_trocr(self) -> list[ModelEntry]:
        """Register the Transformer-OCR baselines (never loaded here)."""
        definitions = (
            (
                "trocr_base",
                "Transformer OCR (base, documents)",
                str(self.config.get("trocr.base_model", "microsoft/trocr-base-handwritten")),
                "TrOCR base fine-tuned on IAM. Preserves letter case, so it is the "
                "strongest general-purpose document recogniser here. ~1.3 GB on first use.",
            ),
            (
                "trocr_handwritten",
                "Transformer OCR (small, documents)",
                str(self.config.get("trocr.handwritten_model", "microsoft/trocr-small-handwritten")),
                "TrOCR small fine-tuned on IAM. Preserves letter case and reads printed "
                "documents well despite the name.",
            ),
            (
                "trocr_printed",
                "Transformer OCR (small, receipts)",
                str(self.config.get("trocr.printed_model", "microsoft/trocr-small-printed")),
                "TrOCR small fine-tuned on SROIE receipts, which are upper case only -- "
                "so it returns UPPER CASE text. Use a documents variant unless you want that.",
            ),
        )

        try:
            import transformers  # noqa: F401

            transformers_available = True
            reason = None
        except ImportError:
            transformers_available = False
            reason = "The `transformers` package is not installed (pip install transformers)."

        return [
            ModelEntry(
                id=model_id,
                name=name,
                family="trocr",
                provenance="pretrained_third_party",
                description=(
                    f"{description} Downloaded weights -- not trained by this project, and "
                    "excluded from the self-supervised comparison."
                ),
                checkpoint=None,
                available=transformers_available,
                unavailable_reason=reason,
                extra={
                    "hf_model_id": hf_id,
                    "downloads_on_first_use": True,
                    # Its authors' training data, not this project's.
                    "trained_on": "SROIE / IAM (by its authors)",
                },
            )
            for model_id, name, hf_id, description in definitions
        ]

    # -- access -------------------------------------------------------------
    def entries(self) -> list[ModelEntry]:
        """All registered models.

        Ordered so the project's own models come first, then the pretrained
        baselines -- the order the comparison table should read in.
        """
        order = {"trained_in_project": 0, "pretrained_third_party": 1}
        return sorted(
            self._entries.values(),
            key=lambda entry: (order.get(entry.provenance, 2), entry.id),
        )

    def entry(self, model_id: str) -> ModelEntry:
        """Metadata for one model.

        Raises:
            ModelNotFound: no such model.
        """
        try:
            return self._entries[model_id]
        except KeyError:
            raise ModelNotFound(
                f"Unknown model {model_id!r}. Available: {', '.join(sorted(self._entries)) or 'none'}"
            ) from None

    def describe_all(self) -> list[dict[str, Any]]:
        """Metadata for every model, including load state when loaded."""
        payload: list[dict[str, Any]] = []
        for entry in self.entries():
            info = entry.to_dict()
            predictor = self._predictors.get(entry.id)
            info["loaded"] = predictor is not None
            if predictor is not None:
                info.update(
                    {k: v for k, v in predictor.describe().items() if k not in ("id", "name")}
                )
            payload.append(info)
        return payload

    def get(self, model_id: str) -> BasePredictor:
        """Return the predictor for *model_id*, loading it on first use.

        Raises:
            ModelNotFound: no such model.
            ModelLoadError: the model exists but could not be loaded.
        """
        entry = self.entry(model_id)

        cached = self._predictors.get(model_id)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._predictors.get(model_id)
            if cached is not None:
                return cached

            if not entry.available:
                raise ModelLoadError(entry.unavailable_reason or f"{model_id} is unavailable")

            predictor = self._build(entry)
            self._predictors[model_id] = predictor

        predictor.warmup()
        return predictor

    def _build(self, entry: ModelEntry) -> BasePredictor:
        if entry.family == "crnn_attention":
            if entry.checkpoint is None:
                raise ModelLoadError(f"{entry.id} has no checkpoint file")
            try:
                return CRNNPredictor.from_checkpoint(
                    entry.checkpoint,
                    device=self.device,
                    model_id=entry.id,
                    model_name=entry.name,
                    provenance=entry.provenance,
                )
            except (FileNotFoundError, RuntimeError) as exc:
                entry.available = False
                entry.unavailable_reason = str(exc)
                raise ModelLoadError(str(exc)) from exc

        if entry.family == "trocr":
            hf_id = str(entry.extra.get("hf_model_id"))
            recognizer = TrOCRRecognizer(
                hf_id,
                self.device,
                max_new_tokens=int(self.config.get("trocr.max_new_tokens", 48)),
                num_beams=int(self.config.get("trocr.num_beams", 1)),
                no_repeat_ngram_size=int(self.config.get("trocr.no_repeat_ngram_size", 3)),
                allow_download=bool(self.config.get("trocr.allow_download", True)),
            )
            try:
                recognizer.ensure_loaded()
            except TrOCRUnavailable as exc:
                entry.available = False
                entry.unavailable_reason = str(exc)
                raise ModelLoadError(str(exc)) from exc

            return TrOCRPredictor(
                recognizer,
                model_id=entry.id,
                model_name=entry.name,
                device=self.device,
                description=entry.description,
            )

        raise ModelLoadError(f"Unsupported model family {entry.family!r} for {entry.id!r}")

    # -- lifecycle ----------------------------------------------------------
    def unload(self, model_id: str) -> bool:
        """Drop a loaded model, freeing its memory.  Returns True if it was loaded."""
        with self._lock:
            predictor = self._predictors.pop(model_id, None)
        if predictor is None:
            return False
        if isinstance(predictor, TrOCRPredictor):
            predictor.recognizer.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded model %r", model_id)
        return True

    def unload_all(self) -> None:
        for model_id in list(self._predictors):
            self.unload(model_id)

    @property
    def loaded_ids(self) -> list[str]:
        return sorted(self._predictors)

    def default_model_id(self) -> str | None:
        """The model to preselect in the UI.

        Preference order: the SimCLR arm (the project's headline result), then
        any other model trained here, then a pretrained baseline.
        """
        available = [entry for entry in self.entries() if entry.available]
        if not available:
            return None
        # `crnn_full` first: the experiment arms are deliberately trained on a
        # crippled 2,000-sample subset to make the self-supervised effect
        # measurable, so they are the wrong default for someone who just wants
        # to read an image.
        for preferred in ("crnn_full", "simclr_finetune", "baseline_scratch"):
            for entry in available:
                if entry.id == preferred:
                    return entry.id
        return available[0].id

    def comparison_candidates(self) -> list[ModelEntry]:
        """Entries that form the controlled self-supervised comparison."""
        return [
            entry
            for entry in self.entries()
            if entry.extra.get("experiment_arm") in ("control", "treatment")
        ]

    def __contains__(self, model_id: object) -> bool:
        return isinstance(model_id, str) and model_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)
