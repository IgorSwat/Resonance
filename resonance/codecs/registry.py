"""The codecs `finalize.codecs` may name. A new codec is one class behind the Codec contract and one line here."""

import importlib

from resonance.codecs.codec import Codec

CODECS = {
    "higgs": "resonance.codecs.higgs:HiggsCodec",
}


def get(name: str) -> type[Codec]:
    if name not in CODECS:
        raise KeyError(f"Unknown codec {name!r}; known: {', '.join(CODECS)}")
    module, _, attribute = CODECS[name].partition(":")
    return getattr(importlib.import_module(module), attribute)
