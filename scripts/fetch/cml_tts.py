"""Download a range of CML-TTS shards from HuggingFace.

    python scripts/fetch/cml_tts.py --lang PL --first 0 --last 3
    python scripts/fetch/cml_tts.py --lang PL --split dev --first 0

CML-TTS is one parquet directory per language in ylacombe/cml-tts, split into train, dev and
test. A shard is the unit scripts/filter/cml_tts.py reads, so it lands in
<output-dir>/<language>/ under its own name rather than being unpacked.

Shard names carry a content hash (train-00000-of-00012-e5cf3a9ceb8d8a0e.parquet) that cannot be
derived from the index, so the repo is listed over the HTTP API first and --first/--last select
from that listing rather than naming files directly.

The repo is public, so no token is needed; one is sent when HF_TOKEN or the CLI cache holds it.
Downloading needs aria2c on PATH. Polish shards are ~450-900 MB each; a partial download resumes
rather than restarting, and an already-downloaded shard is skipped.
"""

import argparse
import pathlib
import shutil
import sys

import requests
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import print_info, print_section, print_test_title
from scripts.fetch.parla_speech_pl import download, token

# language code -> the directory the repo keeps that language under
LANGUAGES = {"NL": "dutch", "FR": "french", "DE": "german", "IT": "italian",
             "PL": "polish", "PT": "portuguese", "ES": "spanish"}
REPO = "ylacombe/cml-tts"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
ROOT = pathlib.Path("data/CML-TTS")
SPLITS = ("train", "dev", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", default="PL", type=str.upper, choices=sorted(LANGUAGES))
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("--first", type=int, required=True, help="first shard id, inclusive")
    parser.add_argument("--last", type=int, help="last shard id, inclusive (default: --first)")
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    language = LANGUAGES[args.lang]
    names = shards(language, args.split)
    last = args.first if args.last is None else args.last
    if last < args.first:
        raise SystemExit(f"--last {last} is before --first {args.first}")
    if not 0 <= args.first or last >= len(names):
        raise SystemExit(f"{language} {args.split} has shards 0-{len(names) - 1}, "
                         f"got {args.first}-{last}")

    directory = args.output_dir / language
    print_test_title(f"Fetching CML-TTS {args.lang}: {args.split} shards {args.first}-{last}")
    print_info("output", directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not shutil.which("aria2c"):
        raise SystemExit("No aria2c on PATH; install it (e.g. `apt install aria2`)")
    bearer = token()

    fetched = 0
    for name in tqdm(names[args.first:last + 1], desc="shards", unit="shard", leave=True):
        print_section(name)
        target = directory / name
        if target.exists():
            print_info("skipped", f"{target} already downloaded "
                                  f"({target.stat().st_size / 1e6:.0f} MB)")
            continue

        download(f"{BASE_URL}/{language}/{name}", target, bearer)
        print_info("downloaded", f"{target.stat().st_size / 1e6:.0f} MB -> {target}")
        fetched += 1

    print_section("Done")
    print_info("shards fetched", fetched)
    print_info("root", directory)


def shards(language, split):
    """The split's shard filenames in index order, read from the repo listing.

    Sorting by name is sorting by index: the shard number is zero-padded and comes before the
    hash that makes the names unguessable.
    """

    response = requests.get(f"{TREE_URL}/{language}", params={"limit": 1000}, timeout=30)
    response.raise_for_status()
    names = sorted(entry["path"].rsplit("/", 1)[-1] for entry in response.json()
                   if entry["path"].rsplit("/", 1)[-1].startswith(f"{split}-"))
    if not names:
        raise SystemExit(f"No {split} shards under {language}/ in {REPO}")
    return names


if __name__ == "__main__":
    main()
