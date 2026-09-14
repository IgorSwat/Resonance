"""Encode the selected CML-TTS clips to Higgs audio tokens.

    python scripts/finalize/cml_tts.py --lang PL --input cml_tts_selected.csv

Materialises the selected subset: the wav next to the tokens, one file each per clip. The clips
are the corpus's own 24 kHz audio, which is already the rate the codec wants, so nothing is
resampled in practice.

The audio is read back out of the parquet shards rather than from a directory of wavs. The filter
writes only a CSV, so the shards stay the single copy of the corpus until a clip is actually
selected, and only the selection is ever written to disk. That costs one scan of the shards for
--lang, which stops as soon as the last selected clip is found.

read_pool lives here for want of a scripts/select/cml_tts.py; a selection script would own it.
"""

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_test_title
from scripts.fetch.cml_tts import LANGUAGES
from scripts.filter.cml_tts import DATASET, ROOT, SEPARATOR, clip_name, decode, rows
from scripts.finalize.libritts import report
from tools.codec.higgs import HiggsCodec, to_codec_rate

PROCESSED = pathlib.Path("data/processed/CML-TTS")
DIALECT = {"delimiter": SEPARATOR, "quotechar": None, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", default="PL", type=str.upper, choices=sorted(LANGUAGES))
    parser.add_argument("--input", type=pathlib.Path,
                        default=pathlib.Path("cml_tts_selected.csv"))
    parser.add_argument("--audio-dir", type=pathlib.Path, default=PROCESSED / "audio")
    parser.add_argument("--codec-dir", type=pathlib.Path, default=PROCESSED / "codecs")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT,
                        help="the parquet shards the clips are read back out of")
    parser.add_argument("--device", help="torch device for the codec; MPS falls back to CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    pool = read_pool(args.input)
    wanted = {name for name, _, _ in pool}
    directory = args.root / LANGUAGES[args.lang]
    shards = sorted(directory.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No {DATASET} shards under {directory}")

    print_test_title(f"Finalizing {DATASET} {args.lang}: {len(wanted)} selected clips")
    print_info("input", args.input)
    print_info("shards", f"{len(shards)} under {directory}")
    print_info("audio", args.audio_dir)
    print_info("codecs", args.codec_dir)

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.codec_dir.mkdir(parents=True, exist_ok=True)
    codec = HiggsCodec(device=args.device)
    print_info("codec", f"{codec.sampling_rate} Hz on {codec.model.device}")

    done, failures, frames, start = 0, 0, 0, time.time()
    for row in rows(shards):
        name = clip_name(row)
        if name not in wanted:
            continue
        wanted.discard(name)
        try:
            audio = decode(row)
            tokens = codec.encode(to_codec_rate(audio, codec.sampling_rate))
            np.save(args.codec_dir / f"{name}.npy", tokens.cpu().numpy().astype(np.int16))
            sf.write(args.audio_dir / f"{name}.wav", audio["audio"], audio["sample_rate"])
        except Exception as error:
            print(f"{Colors.FAIL}{name}: {error}{Colors.ENDC}")
            failures += 1
            continue
        done += 1
        frames += tokens.shape[1]
        if done % 200 == 0:
            print(f"  {done}/{len(pool)} clips, {done / (time.time() - start):.1f}/s", flush=True)
        if not wanted:
            break

    if wanted:
        print(f"{Colors.WARNING}  {len(wanted)} clips not found under {directory}{Colors.ENDC}")
    report(done, failures, frames, time.time() - start, args)


def read_pool(path):
    """The selected CSV, as (name, transcript, speaker) triples."""

    if not path.exists():
        raise SystemExit(f"No input pool at {path}; run scripts/filter/cml_tts.py first")
    with open(path, newline="") as handle:
        return [(row["name"], row["transcription"], row["speaker_id"])
                for row in csv.DictReader(handle, **DIALECT)]


if __name__ == "__main__":
    main()
