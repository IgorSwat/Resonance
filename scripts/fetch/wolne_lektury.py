"""Download a range of WolneLektury shards from HuggingFace.

    python scripts/fetch/wolne_lektury.py --first 0 --last 3

The corpus ships as 381 parquet shards under data/ in
datadriven-company/WolneLektury-TTS-Polish, each holding the audio inline beside its transcript,
speaker and gender columns. A shard is the unit scripts/filter/wolne_lektury.py reads, so it
lands in <output-dir>/ under its own name rather than being unpacked.

The repo also carries four stale shards named train-0000{0,1,2,3}-of-00380.parquet, left over
from an earlier upload; they are not part of the 381 and are never fetched here.

The repo is public, so no token is needed; one is sent when HF_TOKEN or the CLI cache holds it.
Downloading needs aria2c on PATH. Shards are ~430 MB each; a partial download resumes rather
than restarting, and an already-downloaded shard is skipped.
"""

import argparse
import pathlib
import shutil
import sys

from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import print_info, print_section, print_test_title
from scripts.fetch.parla_speech_pl import download, token

BASE_URL = "https://huggingface.co/datasets/datadriven-company/WolneLektury-TTS-Polish/resolve/main/data"
ROOT = pathlib.Path("data/WolneLektury")
SHARDS = 381


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=int, required=True, help="first shard id, inclusive")
    parser.add_argument("--last", type=int, help="last shard id, inclusive (default: --first)")
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    last = args.first if args.last is None else args.last
    if last < args.first:
        raise SystemExit(f"--last {last} is before --first {args.first}")
    if not 0 <= args.first or last >= SHARDS:
        raise SystemExit(f"shard ids run 0-{SHARDS - 1}, got {args.first}-{last}")

    print_test_title(f"Fetching WolneLektury: shards {args.first}-{last}")
    print_info("output", args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("aria2c"):
        raise SystemExit("No aria2c on PATH; install it (e.g. `apt install aria2`)")
    bearer = token()

    fetched = 0
    for shard in tqdm(range(args.first, last + 1), desc="shards", unit="shard", leave=True):
        name = f"train-{shard:05d}-of-{SHARDS:05d}.parquet"
        print_section(name)
        target = args.output_dir / name
        if target.exists():
            print_info("skipped", f"{target} already downloaded "
                                  f"({target.stat().st_size / 1e6:.0f} MB)")
            continue

        download(f"{BASE_URL}/{name}", target, bearer)
        print_info("downloaded", f"{target.stat().st_size / 1e6:.0f} MB -> {target}")
        fetched += 1

    print_section("Done")
    print_info("shards fetched", fetched)
    print_info("root", args.output_dir)


if __name__ == "__main__":
    main()
