"""Copy the selected Emilia clips and encode them to Higgs audio tokens.

    python scripts/finalize/emilia.py --input data/processed/Emilia/emilia_selected.csv

Takes the output of scripts/select/emilia.py and materialises the subset: the mp3 next to the
tokens, one file each per clip. Emilia's mp3s are already 24 kHz, what the codec wants, so they
are copied rather than re-encoded. Tokens are stored int16 as OmniVoice does — the codebook is
1024 wide, so the narrower type halves the files without losing anything.

Clips live in one directory per batch, and every batch under --root is searched; point --root at
a single batch directory to finalize only that one.
"""

import argparse
import pathlib
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_test_title
from scripts.finalize.libritts import report
from scripts.select.emilia import DATASET, ROOT, locate, read_pool
from scripts.select.libritts import load
from tools.codec.higgs import HiggsCodec

PROCESSED = pathlib.Path("data/processed/Emilia")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=pathlib.Path, default=PROCESSED / "emilia_selected.csv")
    parser.add_argument("--audio-dir", type=pathlib.Path, default=PROCESSED / "audio")
    parser.add_argument("--codec-dir", type=pathlib.Path, default=PROCESSED / "codecs")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT,
                        help="searched whole, so all batches under it by default")
    parser.add_argument("--device", help="torch device for the codec; MPS falls back to CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    pool, _ = read_pool(args.input)

    print_test_title(f"Finalizing {DATASET}: {len(pool)} selected clips")
    print_info("input", args.input)
    print_info("audio", args.audio_dir)
    print_info("codecs", args.codec_dir)

    paths = locate(args.root, {name for name, _, _ in pool})
    missing = [name for name, _, _ in pool if name not in paths]
    if missing:
        print(f"{Colors.WARNING}  {len(missing)} clips not found under {args.root}{Colors.ENDC}")

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.codec_dir.mkdir(parents=True, exist_ok=True)
    codec = HiggsCodec(device=args.device)
    print_info("codec", f"{codec.sampling_rate} Hz on {codec.model.device}")

    done, failures, frames, start = 0, 0, 0, time.time()
    for index, (name, _, _) in enumerate(pool, 1):
        if name not in paths:
            continue
        try:
            shutil.copy2(paths[name], args.audio_dir / f"{name}.mp3")
            tokens = codec.encode(load(paths[name]))
            np.save(args.codec_dir / f"{name}.npy", tokens.cpu().numpy().astype(np.int16))
        except Exception as error:
            print(f"{Colors.FAIL}{name}: {error}{Colors.ENDC}")
            failures += 1
            continue
        done += 1
        frames += tokens.shape[1]
        if index % 200 == 0:
            print(f"  {index}/{len(pool)} clips, {index / (time.time() - start):.1f}/s", flush=True)

    report(done, failures, frames, time.time() - start, args)


if __name__ == "__main__":
    main()
