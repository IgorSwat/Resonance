"""
finalize: ClipManifest -> Dataset. Packs the selected clips into the final product.

    runs/<run>/dataset/<run>.csv        the five standard columns
    runs/<run>/dataset/audio.zip        <name>.wav for every selected clip
    runs/<run>/dataset/codecs/<codec>/  <name>.npy, only for the codecs the config names

The selected clips are moved out of clips/, not copied, so the zip holds a part of what the earlier stages
produced and costs no second copy once it is written. That makes this the one irreversible stage: the batches it
drew from are marked consumed by this run, and their manifests point at clips that are gone until
`restore` puts them back.
"""

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np

from resonance.codecs import registry
from resonance.config.schema import Config
from resonance.io import manifest
from resonance.io.audio import read_audio
from resonance.io.manifest import Row
from resonance.log.events import Progress
from resonance.pipeline import gpu, state
from resonance.pipeline.artifacts import ClipManifest, Dataset, Layout
from resonance.pipeline.stage import Context, Stage

COMPRESSION = {"deflate": zipfile.ZIP_DEFLATED, "stored": zipfile.ZIP_STORED}
CONSUMED = "consumed.json"


def encode(task: tuple[str, list[tuple[str, str]]], advance: Callable[[int], None]) -> int:
    """Encode (wav path, npy path) pairs with one codec, in a process pinned to one GPU; returns how many."""

    codec_name, pairs = task
    codec = registry.get(codec_name)()
    for source, target in pairs:
        temporary = f"{target}.tmp.npy"
        np.save(temporary, codec.encode(read_audio(Path(source))))
        os.replace(temporary, target)
        advance(1)
    return len(pairs)


def consumed_marker(layout: Layout, row: Row) -> Path:
    """The marker of the batch a clip came from: clips/<shard>/<batch>/x.wav -> batches/<shard>/<batch>/."""

    shard, batch = Path(row["audio_path"]).parts[1:3]
    return layout.root / "batches" / shard / batch / CONSUMED


class Finalize(Stage):
    name = "finalize"
    scope = "pool"
    consumes = ClipManifest
    produces = Dataset

    def output(self, ctx: Context) -> Dataset:
        return Dataset(ctx.layout.dataset)

    def run(self, source: ClipManifest, ctx: Context) -> Dataset:
        config, dataset = self.config.finalize, ctx.layout.dataset
        rows = manifest.read(source.path)
        audio = dataset / "audio"
        audio.mkdir(parents=True, exist_ok=True)
        manifest.write(dataset / f"{self.config.run.name}.csv", rows, manifest.STANDARD)

        progress = ctx.log.progress(len(rows) * (1 + len(config.codecs) + config.zip))
        for row in rows:
            origin, target = ctx.layout.root / row["audio_path"], audio / f"{row['name']}.wav"
            if not target.exists():         # a rerun after a crash finds some clips already here
                if config.move:
                    marker = consumed_marker(ctx.layout, row)
                    if not marker.exists():
                        marker.write_text(json.dumps({"run": self.config.run.name}))
                    shutil.move(origin, target)
                else:
                    shutil.copy2(origin, target)
            progress.advance()

        for codec in config.codecs:
            folder = dataset / "codecs" / codec
            folder.mkdir(parents=True, exist_ok=True)
            pairs = [(str(audio / f"{row['name']}.wav"), str(folder / f"{row['name']}.npy")) for row in rows
                     if not (folder / f"{row['name']}.npy").exists()]
            progress.advance(len(rows) - len(pairs))
            gpu.map_gpus(encode, [(codec, part) for part in gpu.split(pairs, len(ctx.gpus))], ctx.gpus,
                         self.config.resources.threads_per_worker, progress, self.log_path(ctx))

        size = sum((audio / f"{row['name']}.wav").stat().st_size for row in rows)
        if config.zip:
            size = self._zip(audio, rows, dataset / "audio.zip", COMPRESSION[config.compression], progress)
        progress.flush()
        ctx.summary.update(clips_out=len(rows), hours_out=round(manifest.hours(rows), 3),
                           speakers=len({row["speaker_id"] for row in rows}),
                           audio_gb=round(size / 1024 ** 3, 2), codecs=", ".join(config.codecs) or "none")
        return self.output(ctx)

    @staticmethod
    def _zip(audio: Path, rows: list[Row], target: Path, compression: int, progress: Progress) -> int:
        """Zip audio/ and remove it, but only after the zip has been read back whole."""

        temporary = target.with_name(target.name + ".tmp")
        with zipfile.ZipFile(temporary, "w", compression) as archive:
            for row in rows:
                archive.write(audio / f"{row['name']}.wav", f"{row['name']}.wav")
                progress.advance()
        with zipfile.ZipFile(temporary) as archive:
            damaged = archive.testzip()
            if damaged is not None or len(archive.namelist()) != len(rows):
                raise RuntimeError(f"{temporary} failed verification ({damaged or 'wrong file count'}); "
                                   "audio/ is left in place")
        os.replace(temporary, target)
        shutil.rmtree(audio)
        return target.stat().st_size


def restore(config: Config) -> int:
    """
    Undo finalize's move: put every selected clip back where it came from and drop the dataset.

    Returns how many clips were put back. The dataset folder is removed only once every clip is back in place.
    """

    layout = Layout(config)
    rows = manifest.read(layout.run_dir / "select.csv")
    audio, archive_path = layout.dataset / "audio", layout.dataset / "audio.zip"
    archive = zipfile.ZipFile(archive_path) if archive_path.exists() else None
    restored = 0
    for row in rows:
        origin = layout.root / row["audio_path"]
        if origin.exists():
            continue
        origin.parent.mkdir(parents=True, exist_ok=True)
        loose = audio / f"{row['name']}.wav"
        if loose.exists():
            shutil.move(loose, origin)
        elif archive is not None:
            with archive.open(f"{row['name']}.wav") as packed, open(origin, "wb") as handle:
                shutil.copyfileobj(packed, handle)
        else:
            raise FileNotFoundError(f"{row['name']}.wav is neither in {audio} nor in {archive_path}")
        restored += 1
    if archive is not None:
        archive.close()
    for row in rows:
        marker = consumed_marker(layout, row)
        if marker.exists() and json.loads(marker.read_text())["run"] == config.run.name:
            marker.unlink()
    shutil.rmtree(layout.dataset, ignore_errors=True)
    state.clear(layout.run_dir, "finalize")
    return restored
