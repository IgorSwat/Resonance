"""
The finished runs on Google Drive, through rclone: which are there, and sending one with progress and a hash check.

Layout, as the uploads made by hand before: <path>/ holds audio.zip and <run>.csv, <path>/highlights/ the
highlights files. A run counts as uploaded when audio.zip and <run>.csv are both there.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Callable

from resonance.config.schema import UploadConfig

# Every language of YODAS2-Sidon (its shard prefixes, ISO 639-1; iw is Hebrew) -> its Drive folder name.
LANGUAGE_NAMES = {"aa": "afar", "ab": "abkhazian", "af": "afrikaans", "ak": "akan", "am": "amharic", "ar": "arabic",
                  "as": "assamese", "ay": "aymara", "az": "azerbaijani", "ba": "bashkir", "be": "belarusian",
                  "bg": "bulgarian", "bh": "bihari", "bi": "bislama", "bm": "bambara", "bn": "bengali",
                  "bo": "tibetan", "br": "breton", "bs": "bosnian", "ca": "catalan", "co": "corsican", "cr": "cree",
                  "cs": "czech", "cy": "welsh", "da": "danish", "de": "german", "dz": "dzongkha", "ee": "ewe",
                  "el": "greek", "en": "english", "eo": "esperanto", "es": "spanish", "et": "estonian",
                  "eu": "basque", "fa": "persian", "ff": "fulah", "fi": "finnish", "fj": "fijian", "fo": "faroese",
                  "fr": "french", "fy": "western_frisian", "ga": "irish", "gd": "scottish_gaelic", "gl": "galician",
                  "gn": "guarani", "gu": "gujarati", "ha": "hausa", "hi": "hindi", "ho": "hiri_motu",
                  "hr": "croatian", "ht": "haitian_creole", "hu": "hungarian", "hy": "armenian", "ia": "interlingua",
                  "id": "indonesian", "ie": "interlingue", "ig": "igbo", "ik": "inupiaq", "is": "icelandic",
                  "it": "italian", "iu": "inuktitut", "iw": "hebrew", "ja": "japanese", "jv": "javanese",
                  "ka": "georgian", "ki": "kikuyu", "kk": "kazakh", "kl": "kalaallisut", "km": "khmer",
                  "kn": "kannada", "ko": "korean", "ks": "kashmiri", "ku": "kurdish", "ky": "kyrgyz", "la": "latin",
                  "lb": "luxembourgish", "lg": "ganda", "ln": "lingala", "lo": "lao", "lt": "lithuanian",
                  "lv": "latvian", "mg": "malagasy", "mi": "maori", "mk": "macedonian", "ml": "malayalam",
                  "mn": "mongolian", "mr": "marathi", "ms": "malay", "my": "burmese", "na": "nauru",
                  "nd": "north_ndebele", "ne": "nepali", "nl": "dutch", "no": "norwegian", "nv": "navajo",
                  "oc": "occitan", "om": "oromo", "or": "odia", "pa": "punjabi", "pl": "polish", "ps": "pashto",
                  "pt": "portuguese", "qu": "quechua", "rm": "romansh", "rn": "kirundi", "ro": "romanian",
                  "ru": "russian", "rw": "kinyarwanda", "sa": "sanskrit", "sc": "sardinian", "sd": "sindhi",
                  "sg": "sango", "sh": "serbo_croatian", "si": "sinhala", "sk": "slovak", "sl": "slovenian",
                  "sm": "samoan", "sn": "shona", "so": "somali", "sq": "albanian", "sr": "serbian",
                  "st": "southern_sotho", "su": "sundanese", "sv": "swedish", "sw": "swahili", "ta": "tamil",
                  "te": "telugu", "tg": "tajik", "th": "thai", "ti": "tigrinya", "tk": "turkmen", "tn": "tswana",
                  "to": "tongan", "tr": "turkish", "ts": "tsonga", "tt": "tatar", "ug": "uyghur", "uk": "ukrainian",
                  "ur": "urdu", "uz": "uzbek", "ve": "venda", "vi": "vietnamese", "vo": "volapuk", "wo": "wolof",
                  "xh": "xhosa", "yi": "yiddish", "yo": "yoruba", "zh": "chinese", "zu": "zulu"}


class UploadError(RuntimeError):
    pass


def target(config: UploadConfig, language: str, run: str) -> str:
    """The rclone destination of a run, e.g. gdrive:train/english/yodas2_sidon/en004_100."""

    if language not in LANGUAGE_NAMES:
        raise UploadError(f"{language!r} is not a YODAS2-Sidon language")
    path = config.path.format(language_name=LANGUAGE_NAMES[language], run=run)
    return f"{config.remote}:{path}" if config.remote else path


def _rclone(config: UploadConfig, *args: str) -> list[str]:
    command = ["rclone", *args]
    if config.remote and config.root_folder_id:
        command += ["--drive-root-folder-id", config.root_folder_id]
    return command


def uploaded(config: UploadConfig, language: str, runs: list[str]) -> set[str]:
    """The runs among `runs` whose audio.zip and CSV are already in their Drive folder."""

    parent = target(config, language, "")
    listing = subprocess.run(_rclone(config, "lsf", "-R", "--max-depth", "2", "--files-only", parent),
                             capture_output=True, text=True)
    if listing.returncode:
        if "directory not found" in listing.stderr:
            return set()
        raise UploadError(f"rclone lsf {parent} failed: {listing.stderr.strip()[-500:]}")
    files = set(listing.stdout.split())
    return {run for run in runs if {f"{run}/audio.zip", f"{run}/{run}.csv"} <= files}


def upload(config: UploadConfig, language: str, run: str, dataset: Path, highlights: Path | None,
           log: Callable[[str], None]) -> None:
    """Copy a run's dataset (and highlights) to Drive, logging progress, then check every file's hash."""

    destination = target(config, language, run)
    parts = [(dataset, destination)]
    if config.highlights and highlights is not None and highlights.is_dir():
        parts.append((highlights, f"{destination}/highlights"))
    for source, remote in parts:
        size = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
        log(f"upload {run}: {source.name}/ → {remote} ({size / 1e9:.1f} GB)")
        started = time.time()
        _copy(config, source, remote, run, log)
        elapsed = time.time() - started
        log(f"upload {run}: {source.name}/ sent in {_clock(elapsed)}, {size / 1e6 / max(elapsed, 1):.0f} MB/s average")
        check = subprocess.run(_rclone(config, "check", "--one-way", str(source), remote),
                               capture_output=True, text=True)
        if check.returncode:
            raise UploadError(f"rclone check {source} → {remote} found differences: {check.stderr.strip()[-800:]}")
        log(f"upload {run}: {source.name}/ verified (hashes match)")


def _copy(config: UploadConfig, source: Path, remote: str, run: str, log: Callable[[str], None]) -> None:
    command = _rclone(config, "copy", str(source), remote, "--transfers", str(config.transfers),
                      "--retries", str(config.retries), "--drive-chunk-size", "256M", "--use-json-log",
                      "--stats", f"{config.stats_seconds}s", "--stats-log-level", "NOTICE", "--log-level", "NOTICE")
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    errors = []
    assert process.stderr is not None
    for line in process.stderr:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            errors.append(line.strip())
            continue
        if "stats" in entry:
            line_text = progress_line(entry["stats"])
            if line_text:
                log(f"upload {run}: {line_text}")
        elif entry.get("level") == "error":
            errors.append(entry.get("msg", ""))
    if process.wait():
        raise UploadError(f"rclone copy {source} → {remote} failed: {' | '.join(errors)[-800:]}")


def progress_line(stats: dict) -> str:
    """One progress line from rclone's stats: transferred of total, percent, speed and ETA."""

    total = stats.get("totalBytes") or 0
    if not total:
        return ""
    done = stats.get("bytes", 0)
    eta = stats.get("eta")
    return (f"{done / 1e9:.1f}/{total / 1e9:.1f} GB ({100 * done / total:.0f}%), "
            f"{(stats.get('speed') or 0) / 1e6:.0f} MB/s, ETA {_clock(eta) if eta is not None else '?'}")


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"
