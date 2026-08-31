"""Download and unpack a range of Emilia batches from HuggingFace.

    python scripts/fetch/emilia.py --lang EN --first 1 --last 3
    python scripts/fetch/emilia.py --type yodas --lang EN --first 1 --last 3

Each batch is one WebDataset tar under <Emilia|Emilia-YODAS>/<LANG>/ in amphion/Emilia-Dataset,
and unpacks flat: an mp3 and a JSON sidecar per clip. A batch lands in
<output-dir>/<lang>-b<id>/, the layout scripts/filter/emilia.py expects — data/Emilia for the
classic type, data/Emilia-YODAS for yodas.

YODAS is the raw pre-ASR pool: clip IDs are <LANG>_<youtube_id>_W######, the speaker field keeps
the raw pyannote label, and one video straddles consecutive tars (knowledge/emilia.md §1).

The dataset is gated, so accept its terms on the model page and have a token in HF_TOKEN or
~/.cache/huggingface/token. Downloading needs aria2c on PATH. Batches are ~2 GB each; a
partial download resumes rather than restarting, and an already-unpacked batch is skipped.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_test_title

LANGUAGES = ("EN", "ZH", "DE", "FR", "JA", "KO")
BASE_URL = "https://huggingface.co/datasets/amphion/Emilia-Dataset/resolve/main"
# dataset name in the repo (also the URL path) -> default output directory
TYPES = {"classic": ("Emilia", pathlib.Path("data/Emilia")),
         "yodas": ("Emilia-YODAS", pathlib.Path("data/Emilia-YODAS"))}
ARIA2 = ("--continue=true", "--auto-file-renaming=false", "--max-connection-per-server=8",
         "--split=8", "--max-tries=10", "--retry-wait=30", "--console-log-level=warn")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--type", default="classic", choices=sorted(TYPES),
                        help="classic Emilia, or the raw Emilia-YODAS pool")
    parser.add_argument("--lang", default="EN", type=str.upper, choices=LANGUAGES)
    parser.add_argument("--first", type=int, required=True, help="first batch id, inclusive")
    parser.add_argument("--last", type=int, help="last batch id, inclusive (default: --first)")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        help="default: data/Emilia, or data/Emilia-YODAS for --type yodas")
    parser.add_argument("--keep-archives", action="store_true",
                        help="keep the .tar after unpacking instead of deleting it")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset, default_dir = TYPES[args.type]
    output_dir = args.output_dir or default_dir
    last = args.first if args.last is None else args.last
    if last < args.first:
        raise SystemExit(f"--last {last} is before --first {args.first}")
    ids = range(args.first, last + 1)

    print_test_title(f"Fetching {dataset} {args.lang}: batches {args.first}-{last}")
    print_info("output", output_dir)
    print_info("archives", "kept" if args.keep_archives else "deleted after unpacking")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("aria2c"):
        raise SystemExit("No aria2c on PATH; install it (e.g. `apt install aria2`)")
    bearer = token()

    clips = 0
    for batch in tqdm(ids, desc="batches", unit="batch", position=0, leave=True):
        name = f"{args.lang}-B{batch:06d}"
        directory = output_dir / f"{args.lang.lower()}-b{batch:06d}"
        print_section(name)
        if directory.exists() and any(directory.glob("*.json")):
            print_info("skipped", f"{directory} already holds "
                                  f"{sum(1 for _ in directory.glob('*.json'))} clips")
            continue

        archive = output_dir / f"{name}.tar"
        download(f"{BASE_URL}/{dataset}/{args.lang}/{name}.tar", archive, bearer)
        clips += extract(archive, directory)
        if not args.keep_archives:
            archive.unlink()

    print_section("Done")
    print_info("clips unpacked", clips)
    print_info("root", output_dir)


def token():
    """The HuggingFace token, from the environment or the CLI's cache."""

    for value in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGINGFACE_TOKEN")):
        if value:
            return value.strip()
    cached = pathlib.Path.home() / ".cache/huggingface/token"
    if cached.exists():
        return cached.read_text().strip()
    raise SystemExit("No HuggingFace token; set HF_TOKEN or run `huggingface-cli login`")


def download(url, archive, bearer):
    """Fetch the tar with aria2c: parallel connections, resume, and backoff on a 429."""

    # Fed through stdin rather than argv so the token stays out of the process list.
    request = "\n".join((url, f"  header=Authorization: Bearer {bearer}",
                         f"  dir={archive.parent}", f"  out={archive.name}"))
    if subprocess.run(["aria2c", *ARIA2, "--input-file=-"], input=request, text=True).returncode:
        raise SystemExit(f"{Colors.FAIL}aria2c failed on {url}: on a 401 or 403, accept the "
                         f"dataset terms on huggingface.co and check your token{Colors.ENDC}")
    return archive


def extract(archive, directory):
    """Unpack the tar flat into its batch directory; returns how many clips it held."""

    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"extract {directory.name}", unit="file",
                           position=1, leave=False):
            member.name = pathlib.Path(member.name).name       # flatten any PaxHeader nesting
            tar.extract(member, directory, filter="data")
    clips = sum(1 for _ in directory.glob("*.json"))
    print_info("unpacked", f"{clips} clips -> {directory}")
    return clips


if __name__ == "__main__":
    main()
