"""Encode the selected WolneLektury clips to Higgs audio tokens.

    python scripts/finalize/wolne_lektury.py --input wolne_lektury_selected.csv

Materialises the selected subset: the wav next to the tokens, one file each per clip. The clips
are the corpus's own 24 kHz recordings, which is already the rate the codec wants, so nothing is
resampled in practice.

Unlike the Emilia finalizer there is no separate corpus to copy out of — the filter extracted
these clips from the parquet shards, so --audio-dir defaults to where they already sit and the
audio stays put. Point it somewhere else and each selected clip is *moved* there, which also
leaves the unselected clips behind in --root for deletion.

The selected CSV carries the same columns as ParlaSpeech-PL's, so it is read with that corpus's
pool reader.
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
from scripts.select.libritts import load
from scripts.select.parla_speech_pl import locate, read_pool
from tools.codec.higgs import HiggsCodec, to_codec_rate

DATASET = "WolneLektury"
PROCESSED = pathlib.Path("data/processed/WolneLektury")
ROOT = PROCESSED / "audio"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=pathlib.Path,
                        default=pathlib.Path("wolne_lektury_selected.csv"))
    parser.add_argument("--audio-dir", type=pathlib.Path, default=ROOT,
                        help="where the clips end up; the default is --root, so nothing moves")
    parser.add_argument("--codec-dir", type=pathlib.Path, default=PROCESSED / "codecs")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT,
                        help="where the filter wrote its clips")
    parser.add_argument("--device", help="torch device for the codec; MPS falls back to CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    pool, *_ = read_pool(args.input)

    print_test_title(f"Finalizing {DATASET}: {len(pool)} selected clips")
    print_info("input", args.input)
    print_info("audio", f"{args.audio_dir} (left in place)"
                        if args.audio_dir == args.root else f"{args.root} -> {args.audio_dir}")
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
            # encoded before the move, so a failure leaves the clip where it was
            tokens = codec.encode(to_codec_rate(load(paths[name]), codec.sampling_rate))
            np.save(args.codec_dir / f"{name}.npy", tokens.cpu().numpy().astype(np.int16))
            target = args.audio_dir / f"{name}.wav"
            if paths[name] != target:
                shutil.move(paths[name], target)
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
