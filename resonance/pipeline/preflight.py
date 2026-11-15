"""Checks made before any work starts, so a run fails in its first seconds and not in its third hour."""

import dataclasses
import importlib.util
import json
import shutil
from typing import Callable

from resonance.config.schema import Config
from resonance.pipeline import disk, gpu
from resonance.pipeline.artifacts import Layout, batches

GATED = {"multi_speaker": "pyannote/segmentation-3.0", "speaker_fixup": "pyannote/wespeaker-voxceleb-resnet34-LM"}
MODULES = {
    "segment": ("faster_whisper", "torch", "torchaudio"),
    "multi_speaker": ("pyannote.audio",),
    "nisqa": ("torchmetrics", "librosa"),
    "ctc": ("torchaudio", "uroman"),
    "speaker_drift": ("speechbrain",),
    "transcript_fixup": ("faster_whisper", "verbalis"),
    "speaker_fixup": ("pyannote.audio", "scipy", "librosa"),
    "highlights": ("transformers", "librosa"),
    "higgs": ("transformers", "accelerate"),
}


@dataclasses.dataclass
class Check:
    name: str
    status: str             # ok, warning or fail
    detail: str


def run(config: Config, online: bool = True) -> list[Check]:
    """Every check that applies to the config's pipeline. `online=False` skips the ones that need the network."""

    parts = _parts(config)
    checks = [_modules(parts)]
    if "fetch" in parts:
        found = shutil.which("aria2c")
        checks.append(Check("aria2c", "ok" if found else "fail", found or "not on PATH; apt install aria2"))
    checks.append(_gpus(config))
    if online:
        checks += _hub(config, parts)
    checks += [_consumed(config), _disk(config)]
    return checks


def _parts(config: Config) -> set[str]:
    """The stages, metrics and codecs the config switches on."""

    parts = set(config.pipeline) | (set(config.quality.metrics) if "quality" in config.pipeline else set())
    return parts | (set(config.finalize.codecs) if "finalize" in config.pipeline else set())


def download_models(config: Config, report: Callable[[str], None]) -> None:
    """
    Load every model the pipeline uses once, on the CPU, so each is in the cache before the first batch.

    Without this a stage's worker processes all download the same file at the same moment on a fresh machine.
    """

    parts, steps = _parts(config), []
    models = ({config.segment.whisper} if "segment" in parts else set()) | (
        {config.transcript_fixup.whisper} if "transcript_fixup" in parts else set())
    steps += [(f"whisper {model}", _whisper(model)) for model in sorted(models)]
    if "multi_speaker" in parts:
        steps.append((GATED["multi_speaker"], _pyannote_model))
    if "nisqa" in parts:
        steps.append(("NISQA", _nisqa))
    if "ctc" in parts:
        steps.append(("MMS forced aligner", _mms))
    if "speaker_drift" in parts:
        steps.append(("ECAPA", _ecapa))
    if "speaker_fixup" in parts:
        steps.append((GATED["speaker_fixup"], _wespeaker))
    for codec in sorted(parts & {"higgs"}):
        steps.append((f"codec {codec}", _codec(codec)))
    for name, load in steps:
        report(f"model: {name}")
        load()


def _whisper(name: str) -> Callable[[], object]:
    def load() -> object:
        from faster_whisper import download_model
        return download_model(name)
    return load


def _pyannote_model() -> object:
    from pyannote.audio import Model
    return Model.from_pretrained(GATED["multi_speaker"])


def _nisqa() -> object:
    from torchmetrics.functional.audio.nisqa import _load_nisqa_model
    return _load_nisqa_model()


def _mms() -> object:
    import torchaudio
    return torchaudio.pipelines.MMS_FA.get_model(with_star=False)


def _ecapa() -> object:
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})


def _wespeaker() -> object:
    import torch
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    return PretrainedSpeakerEmbedding(GATED["speaker_fixup"], device=torch.device("cpu"))


def _codec(name: str) -> Callable[[], object]:
    def load() -> object:
        from resonance.codecs import registry
        return registry.get(name)()
    return load


def _modules(parts: set[str]) -> Check:
    wanted = sorted({module for part in parts for module in MODULES.get(part, ())})
    missing = [module for module in wanted if not _importable(module)]
    return Check("python packages", "fail" if missing else "ok",
                 f"missing: {', '.join(missing)}; run scripts/setup.sh (a codec needs `pip install .[<codec>]`)" if missing else f"{len(wanted)} found")


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _gpus(config: Config) -> Check:
    try:
        found = gpu.discover(config.resources.gpus)
    except ValueError as error:
        return Check("GPUs", "fail", str(error))
    if found == [None]:
        return Check("GPUs", "warning", "no NVIDIA GPU; stages will run on whatever device torch finds")
    return Check("GPUs", "ok", f"{len(found)}: {', '.join(str(index) for index in found)}")


def _hub(config: Config, parts: set[str]) -> list[Check]:
    from huggingface_hub import auth_check, get_token

    from resonance.datasets import yodas2_sidon

    token, checks = get_token(), []
    needed = sorted({repo for part, repo in GATED.items() if part in parts})
    if needed:
        checks.append(Check("HF token", "ok" if token else "fail",
                            "found" if token else "none; run scripts/setup.sh or set HF_TOKEN"))
    for repo in needed if token else ():
        try:
            auth_check(repo, token=token)
            checks.append(Check(repo, "ok", "accessible"))
        except Exception as error:
            checks.append(Check(repo, "fail", f"{type(error).__name__}: accept its terms on huggingface.co "
                                              "with the account the token belongs to"))
    if "fetch" in parts:
        for span in config.source.batches:
            shard = f"{config.source.language}{span.shard}"
            try:
                count = len(yodas2_sidon.available(shard))
            except Exception as error:
                checks.append(Check(f"shard {shard}", "fail", str(error)))
                continue
            ok = 0 <= span.first <= span.last < count
            checks.append(Check(f"shard {shard}", "ok" if ok else "fail",
                                f"batches {span.first}-{span.last} of 0-{count - 1}"))
    return checks


def _consumed(config: Config) -> Check:
    """A batch whose clips another run's finalize moved away cannot be used again until that run is restored."""

    layout, taken = Layout(config), {}
    for ref in batches(config):
        marker = layout.batch_dir(ref) / "consumed.json"
        if marker.exists() and (owner := json.loads(marker.read_text())["run"]) != config.run.name:
            taken.setdefault(owner, []).append(ref.key)
    if not taken:
        return Check("batches", "ok", f"{len(batches(config))} requested, none consumed by another run")
    detail = "; ".join(f"{len(keys)} consumed by run {owner} (first: {keys[0]})" for owner, keys in taken.items())
    return Check("batches", "fail", f"{detail}. Restore that run first: scripts/stages/finalize.py --restore")


def _disk(config: Config) -> Check:
    from resonance.pipeline import runner, state

    layout, refs = Layout(config), batches(config)
    batch_stages, _ = runner.validate(runner.resolve(config))
    if not batch_stages:
        return Check("disk", "ok", "no batch stages")
    last = batch_stages[-1].name
    pending = sum(1 for ref in refs if not state.done(layout.batch_dir(ref), last))
    result = disk.estimate(config, pending, len(refs), len(gpu.discover("all")), disk.free_gb(layout.root),
                           disk.measured_gb(layout, refs, last))
    detail = (f"{pending} batches to process need about {result.needed_gb:.0f} GB; "
              f"{result.free_gb:.0f} GB free on {layout.root}")
    if result.needed_gb <= result.free_gb:
        return Check("disk", "ok", detail)
    return Check("disk", "warning", f"{detail}. About {result.fits} batches fit; the run will fill the disk")
