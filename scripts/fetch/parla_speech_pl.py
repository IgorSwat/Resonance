"""Download a range of ParlaSpeech-PL shards from HuggingFace.

    python scripts/fetch/parla_speech_pl.py --first 0 --last 3

The corpus ships as 123 parquet shards under data/ in classla/ParlaSpeech-PL, each holding the
audio inline as a FLAC blob beside its transcript and speaker columns. A shard is the unit
scripts/filter/parla_speech_pl.py reads, so it lands in <output-dir>/ under its own name rather
than being unpacked.

The repo is public, so no token is needed; one is sent when HF_TOKEN or the CLI cache holds it.
Downloading needs aria2c on PATH. Shards are ~500 MB each; a partial download resumes rather
than restarting, and an already-downloaded shard is skipped.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_test_title

BASE_URL = "https://huggingface.co/datasets/classla/ParlaSpeech-PL/resolve/main/data"
ROOT = pathlib.Path("data/ParlaSpeech-PL")
SHARDS = 123
ARIA2 = ("--continue=true", "--auto-file-renaming=false", "--max-connection-per-server=8",
         "--split=8", "--max-tries=10", "--retry-wait=30", "--console-log-level=warn")


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

    print_test_title(f"Fetching ParlaSpeech-PL: shards {args.first}-{last}")
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


def token():
    """The HuggingFace token, if one is around; the dataset is public, so None is fine."""

    for value in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGINGFACE_TOKEN")):
        if value:
            return value.strip()
    cached = pathlib.Path.home() / ".cache/huggingface/token"
    return cached.read_text().strip() if cached.exists() else None


def download(url, target, bearer):
    """Fetch the shard with aria2c: parallel connections, resume, and backoff on a 429."""

    # Fed through stdin rather than argv so the token stays out of the process list.
    request = [url, f"  dir={target.parent}", f"  out={target.name}"]
    if bearer:
        request.insert(1, f"  header=Authorization: Bearer {bearer}")
    if subprocess.run(["aria2c", *ARIA2, "--input-file=-"],
                      input="\n".join(request), text=True).returncode:
        raise SystemExit(f"{Colors.FAIL}aria2c failed on {url}{Colors.ENDC}")
    return target


if __name__ == "__main__":
    main()
