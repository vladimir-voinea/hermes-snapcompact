"""GNU Unifont 8x16 glyph cache for snapcompact.

Glyphs are pre-extracted from ``unifont-17.0.04.bdf`` (GNU GPLv2+ with the font
embedding exception; see NOTICE-Unifont) into a compact binary: ``SCF1`` magic,
u32 glyph count, then per glyph u32 codepoint, u8 width, u8 height, followed by
``(width // 8) * height`` bytes of MSB-first packed bitmap rows.
"""

from __future__ import annotations

import os
import struct
import threading

_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "unifont16.bin")

_LOCK = threading.Lock()
_CACHE: dict[int, tuple[int, bytes]] | None = None

# Fallback chain for codepoints missing from the font: '?' then space.
_FALLBACK_CPS = (0x3F, 0x20)


def _load() -> dict[int, tuple[int, bytes]]:
    global _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        with open(_FONT_PATH, "rb") as fh:
            data = fh.read()
        if data[:4] != b"SCF1":
            raise ValueError("unifont16.bin: bad magic")
        (count,) = struct.unpack_from("<I", data, 4)
        glyphs: dict[int, tuple[int, bytes]] = {}
        off = 8
        for _ in range(count):
            cp, w, h = struct.unpack_from("<IBB", data, off)
            off += 6
            row_bytes = w // 8
            nbytes = row_bytes * h
            glyphs[cp] = (w, data[off:off + nbytes])
            off += nbytes
        _CACHE = glyphs
    return _CACHE


def glyph_width(ch: str) -> int:
    """Cell width in px for ``ch``: 8 for narrow glyphs, 16 for wide (CJK) ones."""
    entry = _load().get(ord(ch))
    return entry[0] if entry else GLYPH_FALLBACK_W


GLYPH_FALLBACK_W = 8


def glyph(ch: str) -> tuple[int, bytes]:
    """Bitmap for ``ch`` as ``(width_px, rows)`` with MSB-first packed rows.

    Missing codepoints fall back to '?' then space; a truly empty font returns a
    blank 8px glyph so rendering never raises.
    """
    glyphs = _load()
    entry = glyphs.get(ord(ch))
    if entry is None:
        for cp in _FALLBACK_CPS:
            entry = glyphs.get(cp)
            if entry is not None:
                break
    if entry is None:
        return 8, bytes(16)
    return entry
