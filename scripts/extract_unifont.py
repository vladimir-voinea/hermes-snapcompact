#!/usr/bin/env python3
"""Rebuild the unifont16.bin glyph cache from a GNU Unifont BDF file.

Input:  unifont-17.0.04.bdf (or the .bdf.gz) from
        https://mirrors.kernel.org/gnu/unifont/
Output: unifont16.bin — SCF1 magic, u32 glyph count, then per glyph:
        u32 codepoint, u8 width, u8 height, (width // 8) * height bytes of
        MSB-first packed bitmap rows. Only 8x16 (narrow) and 16x16 (wide)
        glyphs are kept; wide glyphs span two cells at render time.

Usage:  ./extract_unifont.py unifont-17.0.04.bdf[.gz] > unifont16.bin
"""

import gzip
import struct
import sys


def extract(path: str) -> bytes:
    opener = gzip.open if path.endswith(".gz") else open
    glyphs: dict[int, tuple[int, bytes]] = {}
    cur_cp: int | None = None
    cur_w, cur_h = 16, 16
    rows: list[str] = []
    in_bitmap = False

    def flush() -> None:
        nonlocal cur_cp, rows, in_bitmap
        if cur_cp is not None and cur_h == 16 and rows and in_bitmap:
            wb = cur_w // 8
            if 8 <= cur_w <= 16:
                data = bytearray()
                for r in rows:
                    b = bytes.fromhex(r)
                    data.extend(b[:wb] if len(b) >= wb else b.ljust(wb, b"\x00"))
                glyphs[cur_cp] = (cur_w, bytes(data))
        cur_cp = None
        rows = []
        in_bitmap = False

    with opener(path, "rt") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if line.startswith("STARTCHAR"):
                flush()
            elif line.startswith("ENCODING"):
                try:
                    cur_cp = int(line.split()[1])
                except ValueError:
                    cur_cp = None
            elif line.startswith("BBX"):
                p = line.split()
                cur_w, cur_h = int(p[1]), int(p[2])
            elif line.startswith("BITMAP") and cur_cp is not None:
                rows = []
                in_bitmap = True
            elif in_bitmap and line.startswith("ENDCHAR"):
                flush()
            elif in_bitmap and cur_cp is not None and line \
                    and all(c in "0123456789ABCDEFabcdef" for c in line) and len(rows) < 32:
                rows.append(line)
    flush()

    out = bytearray(b"SCF1")
    out += struct.pack("<I", len(glyphs))
    for cp, (w, data) in sorted(glyphs.items()):
        out += struct.pack("<IBB", cp, w, 16) + data
    return bytes(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.stdout.buffer.write(extract(sys.argv[1]))
