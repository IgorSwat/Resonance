"""
Strict YAML -> Config mappings.

An unknown key, a wrong type or a missing file is an error, never a silent default.
"""

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any

import yaml

from resonance.config.schema import Config


class ConfigError(ValueError):
    pass


def load(path: Path, overrides: typing.Sequence[str] = ()) -> Config:
    """The config at `path`, with `a.b.c=value` overrides applied on top."""

    if not path.is_file():
        raise ConfigError(f"No config file at {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must hold a mapping, got {type(data).__name__}")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ConfigError(f"Override {override!r} is not key=value")
        node = data
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Override {override!r}: {part} is not a section")
        node[leaf] = yaml.safe_load(value)
    return _build(Config, data, "")


def dump(config: Config) -> str:
    return yaml.safe_dump(_plain(dataclasses.asdict(config)), sort_keys=False)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{where or 'config'} must be a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"Unknown key(s) {', '.join(where + key for key in unknown)}; "
                          f"expected {', '.join(sorted(fields))}")
    missing = sorted(name for name, field in fields.items() if name not in data
                     and field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING)
    if missing:
        raise ConfigError(f"Missing required key(s) {', '.join(where + name for name in missing)}")
    return cls(**{key: _convert(hints[key], value, where + key) for key, value in data.items()})


def _convert(hint: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    if dataclasses.is_dataclass(hint):
        return _build(hint, {} if value is None else value, where + ".")
    if origin in (typing.Union, types.UnionType):
        for option in typing.get_args(hint):
            try:
                return _convert(option, value, where)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: {value!r} is not {hint}")
    if hint is type(None):
        if value is None:
            return None
    elif origin is list:
        if isinstance(value, list):
            (item,) = typing.get_args(hint)
            return [_convert(item, entry, f"{where}[{index}]") for index, entry in enumerate(value)]
    elif origin is dict:
        if isinstance(value, dict):
            _, item = typing.get_args(hint)
            return {str(key): _convert(item, entry, f"{where}.{key}") for key, entry in value.items()}
    elif hint is Path:
        if isinstance(value, str):
            return Path(value)
    elif hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif isinstance(value, hint):
        return value
    raise ConfigError(f"{where}: expected {getattr(hint, '__name__', hint)}, got {value!r}")
