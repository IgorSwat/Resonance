"""How Resonance's logs look: colors, banners, key/value lines and tables. Every function returns text."""

import re
from typing import Sequence

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
MAGENTA, CYAN, GREEN, YELLOW, RED = "\033[95m", "\033[96m", "\033[92m", "\033[93m", "\033[91m"
WIDTH = 78
_ANSI = re.compile(r"\033\[[0-9;]*m")


def plain(text: str) -> str:
    return _ANSI.sub("", text)


def banner(title: str) -> str:
    rule = f"{MAGENTA}{BOLD}{'━' * WIDTH}{RESET}"
    return f"\n{rule}\n{MAGENTA}{BOLD}  {title}{RESET}\n{rule}"


def section(title: str) -> str:
    return f"\n{CYAN}{BOLD}▸ {title}{RESET}"


def info(label: str, value: object) -> str:
    return f"  {BOLD}{label}:{RESET} {value}"


def done(text: str) -> str:
    return f"{GREEN}{BOLD}  ✓{RESET} {text}"


def warning(text: str) -> str:
    return f"{YELLOW}{BOLD}  ! {text}{RESET}"


def error(text: str) -> str:
    return f"{RED}{BOLD}  ✗ {text}{RESET}"


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(plain(line[column])) for line in [list(headers), *cells]) for column in range(len(headers))]

    def render(line: Sequence[str]) -> str:
        return "  " + "  ".join(cell + " " * (width - len(plain(cell))) for cell, width in zip(line, widths))

    return "\n".join([f"{BOLD}{render(headers)}{RESET}", "  " + "  ".join("─" * width for width in widths),
                      *(render(line) for line in cells)])


def duration(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{rest:02d}s" if minutes else f"{rest}s"


def summary(fields: dict) -> str:
    """A stage's summary on one line: counts, hours, rejections by reason, wall time."""

    parts = []
    if "clips_in" in fields and "clips_out" in fields:
        parts.append(f"{fields['clips_in']:,} → {fields['clips_out']:,} clips")
    elif "clips_out" in fields:
        parts.append(f"{fields['clips_out']:,} clips")
    if "hours_out" in fields:
        parts.append(f"{fields['hours_out']:.2f} h")
    parts += [f"{key} {value}" for key, value in fields.items()
              if key not in ("clips_in", "clips_out", "hours_out", "seconds", "rejected") and not isinstance(value, dict)]
    if fields.get("rejected"):
        parts.append("rejected: " + ", ".join(f"{reason} {count:,}" for reason, count in
                                              sorted(fields["rejected"].items(), key=lambda item: -item[1])))
    if "seconds" in fields:
        parts.append(duration(fields["seconds"]))
    return " · ".join(parts)
