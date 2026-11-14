"""
YODAS2-Sidon on HuggingFace: where its batches are and how one recording is read.

The corpus is one directory per language shard in sarulab-speech/yodas2_sidon, named <lang><NNN> (pl000, en001),
each holding a numbered run of train-XXXXX.tar.gz batches of about 3 GB. A batch unpacks flat: one <n>.flac per
video beside its <n>.metadata.json.
"""

import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

DATASET = "YODAS2-Sidon"
REPO = "sarulab-speech/yodas2_sidon"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
ARIA2 = ("--continue=true", "--auto-file-renaming=false", "--max-connection-per-server=8",
         "--split=8", "--max-tries=10", "--retry-wait=30", "--console-log-level=warn", "--summary-interval=0")


def available(shard: str) -> list[str]:
    """The shard's batch file names in order; raises when the shard does not exist."""

    response = requests.get(f"{TREE_URL}/{shard}", params={"limit": 1000}, timeout=30)
    if response.status_code == 404:
        raise ValueError(f"No shard {shard} in {REPO}")
    response.raise_for_status()
    return sorted(entry["path"].rsplit("/", 1)[-1] for entry in response.json()
                  if entry["path"].endswith(".tar.gz"))


def download(shard: str, batch: str, archive: Path, token: str | None) -> None:
    """Fetch one batch archive with aria2c: parallel connections, resume, and backoff on a 429."""

    # Fed through stdin rather than argv so the token stays out of the process list.
    request = [f"{BASE_URL}/{shard}/{batch}.tar.gz", f"  dir={archive.parent}", f"  out={archive.name}"]
    if token:
        request.insert(1, f"  header=Authorization: Bearer {token}")
    done = subprocess.run(["aria2c", *ARIA2, "--input-file=-"], input="\n".join(request), text=True,
                          capture_output=True)
    if done.returncode:
        raise RuntimeError(f"aria2c failed on {shard}/{batch}: {(done.stderr or done.stdout).strip()[-500:]}")


def extract(archive: Path, directory: Path) -> int:
    """Unpack the archive flat into `directory`; returns how many recordings it held."""

    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            member.name = Path(member.name).name        # flatten any PaxHeader nesting
            tar.extract(member, directory, filter="data")
    return len(recordings(directory))


def recordings(directory: Path) -> list[Path]:
    """The metadata files of a raw batch, one per recording, in order."""

    return sorted(directory.glob("*.metadata.json"))


def read_recording(meta_path: Path, margin: float) -> tuple[dict, np.ndarray, int, float, float] | None:
    """
    A recording's captioned span widened by `margin`: (meta, audio, rate, offset, duration), or None when it
    has no captions or cannot be read.

    Only the span between the first caption's start and the last caption's end is read: the rest of the video
    holds nothing the corpus labelled as speech. Caption spans can overrun the file, so the end is clamped.
    """

    flac = meta_path.with_name(meta_path.name.replace(".metadata.json", ".flac"))
    meta = json.loads(meta_path.read_text())
    spans = meta.get("utterances") or {}
    if not spans.get("start"):
        return None
    with sf.SoundFile(flac) as handle:
        rate, duration = handle.samplerate, handle.frames / handle.samplerate
        begin = max(0.0, min(spans["start"]) - margin)
        end = min(duration, max(spans["end"]) + margin)
        if end <= begin:
            return None
        handle.seek(int(begin * rate))
        audio = handle.read(int((end - begin) * rate), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return meta, audio, rate, begin, duration
