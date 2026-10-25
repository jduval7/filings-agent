"""Text normalization for verbatim checks.

Models re-type punctuation: gpt-oss writes "ten‑for‑one" with U+2011 non-breaking hyphens where
the filing has "ten\\-for\\-one" (an escaped ASCII hyphen in the markdown). Those are the same
words; a verifier that rejects them measures typography, not fabrication. Normalizing dashes,
quotes, spaces and markdown escapes keeps the check strict about WORDS.
"""

from __future__ import annotations

import re
import unicodedata

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = dict.fromkeys(map(ord, "‘’‚‛′"), "'") | dict.fromkeys(map(ord, "“”„‟″"), '"')


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_QUOTES)
    text = text.replace("\\", "")  # markdown escapes: "Inc\\." -> "Inc."
    text = re.sub(r"[*_|#]", " ", text)  # emphasis, table pipes, stray heading markers ("## ")
    text = re.sub(r"[\"']", "", text)
    text = re.sub(r"\s*-\s*", "-", text)  # "20 \-for\-1" (filing) == "20-for-1" (model)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)  # "Contracts \." -> "Contracts."
    text = re.sub(r"\(\s+", "(", text)
    return re.sub(r"\s+", " ", text).strip().casefold()
