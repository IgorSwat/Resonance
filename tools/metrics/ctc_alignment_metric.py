import re
from typing import override

import numpy as np
import torch
import torchaudio
from uroman import Uroman

from tools.metrics.metric import Metric

BUNDLE = torchaudio.pipelines.MMS_FA
DICTIONARY = BUNDLE.get_dict(star=None)
BLANK = 0
FIELDS = ("loss", "maxrate", "gap_speech", "trail_speech")


class CtcAlignmentMetric(Metric):
    """
    Transcript/audio agreement from a single CTC forced-alignment pass (MMS_FA, 1130 languages).

    Scores the transcript it is given instead of decoding a new one, and reports two defects
    that need different evidence:

    - **missing words** — the audio stops (or starts) partway through its transcript. The
      aligner runs out of frames for the unspoken tokens and crams them against MMS's 50 Hz
      ceiling, so `maxrate` spikes while `loss` rises.
    - **redundant speech** — the audio says more than the transcript does. Every token still
      aligns comfortably, so `loss` barely moves; the evidence is unaligned frames whose
      posterior is not blank. A real pause is blank, unspoken-in-transcript speech is not.

    Transcript normalization begins with uroman over non-ASCII text (disable with
    `uroman=False`): MMS's alphabet is lowercase Latin plus apostrophe, and normalize() drops
    every other character, so "Kålsva" would otherwise align as the fragments "k lsva".
    """

    def __init__(self, device=None, windows=(4, 6, 8), uroman=True):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.windows = windows
        self.uroman = uroman
        self._model = None
        self._uroman = None

    @override
    def evaluate(self, audio, transcript=None):
        """
        The four alignment scores; higher is worse on every one.
        """

        if not transcript:
            raise ValueError("CtcAlignmentMetric needs a transcript")
        tokens = tokenize(self._romanize(transcript))

        emission, duration = self._emission(audio)
        target = torch.tensor([[t for token in tokens for t in token]])
        loss = ctc_loss(emission, target)

        try:
            aligned, scores = torchaudio.functional.forced_align(emission, target, blank=BLANK)
        except RuntimeError:
            # The transcript needs more frames than the audio has: missing words in the extreme.
            # ctc_loss(zero_infinity=True) scores that 0.0 — the best value there is — so the
            # loss computed above must not be reported for this path.
            return {"loss": float("inf"), "maxrate": float(BUNDLE.sample_rate),
                    "gap_speech": 0.0, "trail_speech": 0.0}

        spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())
        step = duration / emission.shape[1]
        words_spans, cursor = [], 0
        for token in tokens:
            part = spans[cursor : cursor + len(token)]
            cursor += len(token)
            words_spans.append({"start": part[0].start * step, "end": part[-1].end * step,
                                "nch": len(token)})

        speech = (1 - emission[0, :, BLANK].exp()).numpy()
        starts = [w["start"] for w in words_spans]
        ends = [w["end"] for w in words_spans]
        return {
            "loss": loss,
            "maxrate": max_char_rate(words_spans, self.windows),
            "gap_speech": max((speech_mass(speech, step, end, start)
                               for start, end in zip(starts[1:], ends[:-1])), default=0.0),
            "trail_speech": speech_mass(speech, step, ends[-1], duration),
        }

    @override
    def validate(self, audio, lbound=None, rbound=None, transcript=None):
        """
        Pass only when neither defect is reported.

        `rbound` overrides individual thresholds; `lbound` is unused, since every score here is
        an upper bound. Thresholds are corpus-specific — see DEFAULT_RBOUND.
        """

        bounds = dict(DEFAULT_RBOUND)
        if rbound:
            unknown = set(rbound) - set(bounds)
            if unknown:
                raise KeyError(f"Unknown alignment scores: {sorted(unknown)}; expected {list(bounds)}")
            bounds.update(rbound)

        try:
            scores = self.evaluate(audio, transcript)
        except Exception as error:
            print(f"CtcAlignmentMetric failed: {error}")
            return False
        return not (missing_words(scores, bounds) or redundant_speech(scores, bounds))

    def losses(self, audio, transcripts):
        """
        The CTC loss of each candidate transcript, over one shared emission.

        For choosing between spellings of the same audio — a year read as "nineteen ninety-nine"
        or as "one thousand, nine hundred and ninety-nine" — where only the target changes and
        re-running the acoustic model for each candidate would be waste. A candidate with
        nothing alignable in it scores infinity rather than raising.
        """

        emission, _ = self._emission(audio)
        scored = []
        for transcript in transcripts:
            try:
                tokens = tokenize(self._romanize(transcript))
            except ValueError:
                scored.append(float("inf"))
                continue
            scored.append(ctc_loss(emission, torch.tensor([[t for tok in tokens for t in tok]])))
        return scored

    def _romanize(self, transcript):
        """
        uroman over non-ASCII text, MMS's own recipe; ASCII pays nothing.
        """

        if not self.uroman or transcript.isascii():
            return transcript
        if self._uroman is None:
            self._uroman = Uroman()
        return self._uroman.romanize_string(transcript)

    def _emission(self, audio):
        if self._model is None:
            self._model = BUNDLE.get_model(with_star=False).to(self.device).eval()
        y = np.asarray(audio["audio"], dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        waveform = torch.from_numpy(y)[None]
        if audio["sample_rate"] != BUNDLE.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, audio["sample_rate"], BUNDLE.sample_rate)
        with torch.inference_mode():
            emission, _ = self._model(waveform.to(self.device))
        return emission.cpu(), waveform.shape[1] / BUNDLE.sample_rate


def missing_words(scores, bounds):
    """
    Two ways in. Both terms are required for the ordinary case, since loss alone ranks spoken
    numerals rather than defects. But maxrate cannot corroborate anything for a transcript
    shorter than the windows it averages over — max_char_rate has no window to fill and returns
    0.0 — so a loss beyond max_loss counts as a defect on its own.
    """

    return (scores["loss"] > bounds["max_loss"]
            or (scores["loss"] > bounds["loss"] and scores["maxrate"] > bounds["maxrate"]))


def redundant_speech(scores, bounds):
    return (scores["gap_speech"] > bounds["gap_speech"]
            or scores["trail_speech"] > bounds["trail_speech"])


def normalize(text):
    """
    MMS romanization: lowercase Latin plus apostrophe. '-' is the blank symbol, never a target.

    Every other character is dropped, so non-Latin scripts and diacritics must be romanized
    (uroman, see _romanize) before they get here — "Kålsva" would otherwise shatter into
    "k lsva" and "Beyoncé" lose its last letter.
    """

    text = text.lower().replace("’", "'")
    return [w for w in re.sub(r"\s+", " ", re.sub(r"[^a-z']", " ", text)).strip().split(" ")
            if any(c.isalpha() for c in w)]


def tokenize(transcript):
    """
    MMS token ids per word, dropping characters the dictionary cannot represent.
    """

    tokens = [[DICTIONARY[c] for c in word if DICTIONARY.get(c, BLANK) != BLANK]
              for word in normalize(transcript)]
    tokens = [t for t in tokens if t]
    if not tokens:
        raise ValueError(f"No alignable characters in transcript: {transcript!r}")
    return tokens


def ctc_loss(emission, target):
    """
    Per-token CTC loss of one target against one emission.
    """

    return torch.nn.functional.ctc_loss(
        emission.transpose(0, 1), target,
        torch.tensor([emission.shape[1]]), torch.tensor([target.shape[1]]),
        blank=BLANK, reduction="sum", zero_infinity=True,
    ).item() / target.shape[1]


def max_char_rate(word_spans, windows):
    """
    Highest realized character rate over a short window of words, in characters per second.
    """

    return max((sum(w["nch"] for w in word_spans[start : start + n])
                / max(word_spans[start + n - 1]["end"] - word_spans[start]["start"], 1e-3)
                for n in windows for start in range(len(word_spans) - n + 1)), default=0.0)


def speech_mass(speech, step, start, end):
    """
    Seconds of non-blank posterior inside an unaligned stretch.
    """

    lo, hi = int(round(start / step)), int(round(end / step))
    lo, hi = max(0, min(lo, len(speech))), max(0, min(hi, len(speech)))

    return float(speech[lo:hi].sum() * step) if hi > lo else 0.0


DEFAULT_RBOUND = {"loss": 1.0, "max_loss": 5.0, "maxrate": 30.0,
                  "gap_speech": 0.06, "trail_speech": 0.05}
