"""Scores every utterance of a filtered data CSV by how far the student trails the teacher on it, for data selection.

    cd <Echo checkout> && python <Resonance>/scripts/score/yodas2_sidon_dce.py --model-config echo/configs/p2_final.json \
        --teacher-config echo/configs/teacher_standard.json \
        --input <filtered.csv> --audio-dir <clips> --output <scored.csv>

The models are Echo's, so this runs from an Echo checkout: the echo package must be importable and the config paths
resolve against it. scripts/select/yodas2_sidon_dce.py then selects from the scored CSV.

Reads a CSV as Resonance's filter scripts write it (dataset|name|transcription|speaker_id|language, one <name>.wav per
row under --audio-dir), and writes the same rows with one more column, dce_score. The clips are encoded with the
teacher's own audio codec on the fly, so no codec files are needed.

At three states of the two-stage s8 sampler - s8_f0 (codebooks 0-1 fully masked), s8_f7 (averaged over --seeds mask
draws) and s8_polish - it takes both models' single-step CE on the conditional branch, with the prefix fixed at the
eval ratio. The score is the mean over the three of max(student CE - teacher CE, 0); a high one marks an utterance
unlike what the student was trained on. --details also writes both CEs and the KL at every state.

Masks are seeded per utterance rather than per batch position, so a score does not depend on how the utterances
happened to be batched.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from echo import build_model
from echo.data.constants import IGNORE_INDEX
from echo.data.inputs import build_model_inputs
from echo.data.loader import collate
from echo.data.preprocessor import EchoPreprocessor
from echo.eval.constants import POLISH_STATE, PREFIX_RATIO, TRAJECTORY
from echo.masking.audio_masking import _draw_target_mask, codebook_weights
from echo.model.model import EchoModel
from echo.utils.distributed_utils import setup
from echo.utils.seed_utils import seed_of

SEED = 7
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
# Resonance's data CSV dialect: '|'-separated, never quoted.
DIALECT = {"delimiter": "|", "quotechar": None, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}
SCORE = "dce_score"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-config", type=Path, required=True, help="Student's .json config")
    parser.add_argument("--teacher-config", type=Path, required=True, help="Teacher's .json config")
    parser.add_argument("--input", type=Path, required=True, help="Filtered data .csv, as Resonance's filter scripts write it")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Where the CSV's clips are, as <name>.wav")
    parser.add_argument("--output", type=Path, required=True, help=f"The input's rows with a {SCORE} column")
    parser.add_argument("--details", type=Path, help="Optional .csv of both CEs and the KL at every state")
    parser.add_argument("--seeds", type=int, default=5, help="Mask draws averaged at s8_f7")
    parser.add_argument("--max-tokens", type=int, default=24_000, help="Padded tokens per batch (rows x longest row)")
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16",
                        help="Weights' dtype; the losses are always taken in float32. bfloat16 is 1.6x faster and ranks as float32 does (Spearman 0.999)")
    args = parser.parse_args()

    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle, **DIALECT))

    device = setup()
    student = build_model(args.model_config, device=device, dtype=DTYPES[args.dtype]).eval()
    teacher = build_model(args.teacher_config, device=device, dtype=DTYPES[args.dtype]).eval()
    preprocessor = EchoPreprocessor(prefix_min_ratio=PREFIX_RATIO, prefix_max_ratio=PREFIX_RATIO)

    samples = [_sample(row, args.audio_dir, teacher, preprocessor) for row in rows]
    print(f"encoded {len(samples)} utterances", flush=True)

    # (name, rates, polish, draws): f0 and polish mask everything they score, so one draw is exact.
    points = [
        ("f0", TRAJECTORY["s8_f0"][0], False, 1),
        ("f7", TRAJECTORY["s8_f7"][0], False, args.seeds),
        ("polish", POLISH_STATE, True, 1),
    ]
    weights = {polish: torch.tensor(codebook_weights(float(polish))).to(device) for polish in (False, True)}
    details: Dict[int, List[float]] = {}

    for indices in _batches(samples, args.max_tokens):
        batch = collate([samples[index] for index in indices], preprocessor, disable_prefix_drop=True, disable_masking=True)
        scores: Dict[str, torch.Tensor] = {}

        for point_index, (point, rates, polish, draws) in enumerate(points):
            total = torch.zeros(3, len(indices), device=device)
            for draw in range(draws):
                masks = _draw_masks(batch, rates, indices, point_index, draw)
                inputs = build_model_inputs(batch, masks, [polish] * len(indices), device=device)
                # The models would otherwise take their own CE over every position of every codebook.
                labels = inputs.pop("labels")
                with torch.inference_mode():
                    total += _score(student, teacher, inputs, labels, weights[polish])
            scores[point] = (total / draws).cpu()

        for row, index in enumerate(indices):
            details[index] = [float(value) for point, *_ in points for value in scores[point][:, row]]
        print(f"{len(details)}/{len(samples)} utterances", flush=True)

    names = [f"{key}_{point}" for point, *_ in points for key in ("ce_s", "ce_t", "kl")]
    for index, row in enumerate(rows):
        # (ce_s, ce_t, kl) per point: the score is the mean of the clamped CE gaps.
        values = details[index]
        gaps = [max(values[at] - values[at + 1], 0.0) for at in range(0, len(values), 3)]
        row[SCORE] = f"{sum(gaps) / len(gaps):.4f}"

    _write(args.output, rows)
    if args.details:
        _write(args.details, [{"name": row["name"], **{key: f"{value:.4f}" for key, value in zip(names, details[index])}}
                              for index, row in enumerate(rows)])


def _sample(row: Dict[str, str], audio_dir: Path, teacher: EchoModel, preprocessor: EchoPreprocessor) -> Dict[str, Any]:
    """
    One row as the preprocessor takes it, its audio encoded to codec tokens by the teacher's codec. 'length' is its
    whole sequence - prompt, text and audio - which the batching packs by.
    """

    audio, rate = sf.read(audio_dir / f"{row['name']}.wav", dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.mean(axis=1))
    if rate != teacher.sampling_rate:
        waveform = torchaudio.functional.resample(waveform, rate, teacher.sampling_rate)

    with torch.inference_mode():
        features = teacher.feature_extractor(raw_audio=waveform.numpy(), sampling_rate=teacher.sampling_rate, return_tensors="pt")
        tokens = teacher.audio_tokenizer.encode(features["input_values"].to(teacher.device)).audio_codes[0].cpu()

    text, language = row["transcription"], row["language"]
    length = tokens.size(-1) + preprocessor.text_tokens(text).numel() + preprocessor.prompt_tokens(language).numel()
    return {"audio_tokens": tokens.long(), "text": text, "speaker_id": row["speaker_id"], "language": language, "length": length}


def _batches(samples: List[Dict[str, Any]], max_tokens: int) -> List[List[int]]:
    """
    Utterances sorted by length and grouped while rows x longest row stays within 'max_tokens', so short utterances
    batch wide and long ones narrow, with little padding in either.
    """

    order = sorted(range(len(samples)), key=lambda index: samples[index]["length"])
    batches: List[List[int]] = [[]]
    for index in order:
        if batches[-1] and (len(batches[-1]) + 1) * samples[index]["length"] > max_tokens:
            batches.append([])
        batches[-1].append(index)
    return batches


def _draw_masks(
    batch: Dict[str, List],
    rates: Sequence[float],   # The masking rate of every codebook
    indices: List[int],       # Index of every row in the input, which seeds its draw
    point: int,
    draw: int,
) -> List[torch.Tensor]:
    """
    One [C, T] target mask per row, as echo.eval.single_step draws them, seeded by the utterance instead of its place
    in the batch.

    The draw sees the utterance's own frames only: drawn over the padded row, the same seed gives a different mask
    whenever the batch's longest row changes.
    """

    masks = []
    for row, index in enumerate(indices):
        torch.manual_seed(seed_of(SEED, index, point, draw))
        audio = batch["audio_tokens"][row]
        prefix_frames = batch["prefix_length"][row] - batch["prompt_tokens"][row].numel() - batch["text_tokens"][row].numel()
        frames = prefix_frames + batch["target_length"][row]
        mask = torch.zeros_like(audio, dtype=torch.bool)
        mask[:, :frames] = _draw_target_mask(torch.tensor(rates), audio[:, :frames], prefix_frames)
        masks.append(mask)
    return masks


def _score(
    student: EchoModel,
    teacher: EchoModel,
    inputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,     # [B, C, L], IGNORE_INDEX off the scored positions
    weights: torch.Tensor,    # [C] weight of every codebook
) -> torch.Tensor:
    """
    [3, B]: student CE, teacher CE and KL(teacher || student) of every row, each reduced as the eval's losses are - a
    mean over the scored positions of every codebook, then a weighted sum over the codebooks.

    Only the scored positions leave the logits: the softmaxes run over those alone rather than over every position
    of every codebook.
    """

    rows, codebooks, positions = (labels != IGNORE_INDEX).nonzero(as_tuple=True)
    targets = labels[rows, codebooks, positions].unsqueeze(1)

    student_log_probs = F.log_softmax(student(**inputs).logits[rows, codebooks, positions].float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher(**inputs).logits[rows, codebooks, positions].float(), dim=-1)

    per_token = torch.stack([
        -student_log_probs.gather(1, targets).squeeze(1),
        -teacher_log_probs.gather(1, targets).squeeze(1),
        (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(dim=-1),
    ])

    batch, count = labels.shape[:2]
    slot = rows * count + codebooks
    counts = torch.bincount(slot, minlength=batch * count).view(batch, count)
    sums = torch.zeros(3, batch * count, device=labels.device).index_add_(1, slot, per_token).view(3, batch, count)
    means = sums / counts.clamp(min=1)

    return (means * (counts > 0) * weights).sum(dim=-1)


def _write(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), **DIALECT)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
