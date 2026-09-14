"""Download and unpack a range of YODAS2-Sidon batches from HuggingFace.

    python scripts/fetch/yodas2_sidon.py --lang pl --shard 000 --first 0 --last 3

The corpus is one directory per language shard in sarulab-speech/yodas2_sidon, named <lang><NNN>
— pl000, es104, it101 — and each holds a numbered run of train-XXXXX.tar.gz batches. --lang and
--shard pick the directory, --first and --last a range of batches inside it.

Shards are not one per language and their batch counts differ: Polish has only pl000 with 38
batches, Spanish has 8 shards of 130-140, English has 34. So a batch lands in
<output-dir>/<lang>/<shard>/train-XXXXX/, nested by shard — every shard numbers its batches from
zero, and without that nesting pl000 and pl001 would both want train-00000.

A batch unpacks flat: one <n>.flac per video beside its <n>.metadata.json, which is the layout
scripts/filter/yodas2_sidon.py expects. Point that script's --root at <output-dir>/<lang> to
filter every shard of a language at once, since it searches recursively.

The repo is public, so no token is needed; one is sent when HF_TOKEN or the CLI cache holds it.
Downloading needs aria2c on PATH. Batches are ~3 GB each; a partial download resumes rather than
restarting, and an already-unpacked batch is skipped.
"""

import argparse
import pathlib
import shutil
import sys

import requests
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import print_info, print_section, print_test_title
from scripts.fetch.emilia import extract
from scripts.fetch.parla_speech_pl import download, token

REPO = "sarulab-speech/yodas2_sidon"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
ROOT = pathlib.Path("data/YODAS2-sidon")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", required=True, type=str.lower,
                        help="language code, e.g. pl, es, it")
    parser.add_argument("--shard", default="000",
                        help="shard id within the language, e.g. 000 or 104")
    parser.add_argument("--first", type=int, required=True, help="first batch id, inclusive")
    parser.add_argument("--last", type=int, help="last batch id, inclusive (default: --first)")
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT)
    parser.add_argument("--keep-archives", action="store_true",
                        help="keep the .tar.gz after unpacking instead of deleting it")
    return parser.parse_args()


def main():
    args = parse_args()
    config = f"{args.lang}{args.shard}"
    available = batches(config, args.lang)
    last = args.first if args.last is None else args.last
    if last < args.first:
        raise SystemExit(f"--last {last} is before --first {args.first}")
    if not 0 <= args.first or last >= len(available):
        raise SystemExit(f"{config} has batches 0-{len(available) - 1}, got {args.first}-{last}")

    directory = args.output_dir / args.lang / args.shard
    print_test_title(f"Fetching {config}: batches {args.first}-{last} of {len(available)}")
    print_info("output", directory)
    print_info("archives", "kept" if args.keep_archives else "deleted after unpacking")
    directory.mkdir(parents=True, exist_ok=True)
    if not shutil.which("aria2c"):
        raise SystemExit("No aria2c on PATH; install it (e.g. `apt install aria2`)")
    bearer = token()

    clips = 0
    for batch in tqdm(range(args.first, last + 1), desc="batches", unit="batch",
                      position=0, leave=True):
        name = f"train-{batch:05d}"
        print_section(f"{config}/{name}")
        unpacked = directory / name
        if unpacked.exists() and any(unpacked.glob("*.metadata.json")):
            print_info("skipped", f"{unpacked} already holds "
                                  f"{sum(1 for _ in unpacked.glob('*.metadata.json'))} recordings")
            continue

        archive = directory / f"{name}.tar.gz"
        download(f"{BASE_URL}/{config}/{name}.tar.gz", archive, bearer)
        clips += extract(archive, unpacked)
        if not args.keep_archives:
            archive.unlink()

    print_section("Done")
    print_info("recordings unpacked", clips)
    print_info("root", directory)


def batches(config, language):
    """The shard's batch filenames in order; also what says whether the shard exists at all."""

    response = requests.get(f"{TREE_URL}/{config}", params={"limit": 1000}, timeout=30)
    if response.status_code == 404:
        raise SystemExit(f"No shard {config} in {REPO}; {language} has "
                         f"{', '.join(shards(language)) or 'none'}")
    response.raise_for_status()
    found = sorted(entry["path"].rsplit("/", 1)[-1] for entry in response.json()
                   if entry["path"].endswith(".tar.gz"))
    if not found:
        raise SystemExit(f"No batches under {config}/ in {REPO}")
    return found


def shards(language):
    """Every shard id the repo holds for one language, for the error message above."""

    response = requests.get(TREE_URL, params={"limit": 1000}, timeout=30)
    response.raise_for_status()
    return sorted(entry["path"] for entry in response.json()
                  if entry["type"] == "directory" and entry["path"].startswith(language)
                  and entry["path"][len(language):].isdigit())


if __name__ == "__main__":
    main()
