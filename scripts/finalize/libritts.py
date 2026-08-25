"""Copy the selected LibriTTS clips and encode them to Higgs audio tokens.

    python scripts/finalize/libritts.py --input data/processed/LibriTTS/libritts_selected.csv

Takes the output of scripts/select/libritts.py and materialises the subset: the wav next to
the tokens, one file each per clip. Tokens are stored int16 as OmniVoice does — the codebook
is 1024 wide, so the narrower type halves the files without losing anything.
"""

import argparse
import pathlib
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_separator, print_test_title
from scripts.select.libritts import DATASET, ROOT, load, locate, read_pool
from tools.codec.higgs import HiggsCodec

PROCESSED = pathlib.Path("data/processed/LibriTTS")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=pathlib.Path, default=PROCESSED / "libritts_selected.csv")
    parser.add_argument("--audio-dir", type=pathlib.Path, default=PROCESSED / "audio")
    parser.add_argument("--codec-dir", type=pathlib.Path, default=PROCESSED / "codecs")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--device", help="torch device for the codec; MPS falls back to CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    pool = read_pool(args.input)

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
            shutil.copy2(paths[name], args.audio_dir / f"{name}.wav")
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


def report(done, failures, frames, elapsed, args):
    print_section("Written")
    print_info("clips", done)
    if failures:
        print(f"  {Colors.FAIL}failed{Colors.ENDC}: {failures}")
    print_info("tokens", f"{frames} frames ({frames / 25 / 3600:.2f} h at 25 Hz)")
    print_info("audio", args.audio_dir)
    print_info("codecs", args.codec_dir)
    print_separator()
    print_info("elapsed", f"{elapsed / 60:.1f} min ({done / max(elapsed, 1e-9):.1f} clips/s)")


if __name__ == "__main__":
    main()
