"""Text -> PNG frame rasterizer for snapcompact.

Deterministic, local, no network: normalized text is word-wrapped to the shape's
column count, paginated into frames that hug their rows, and rasterized as 1-bit
black-on-white PNGs from 8x16 Unifont glyph bitmaps. Same text + shape => same
bytes, so repeated compactions are stable.
"""

from __future__ import annotations

import io
import re

from . import glyphs
from .shapes import GLYPH_H, Shape

_ANSI_RE = re.compile(
    r"\x1b(?:\[[0-9;?]*[A-Za-z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
)
_PARA_BLOCK = "\u2588" * 16  # full-block paragraph separator line


def normalize(text: str) -> str:
    """Strip ANSI, expand tabs, fold blank-line runs into a visible block line."""
    text = _ANSI_RE.sub("", text).replace("\r", "").replace("\t", "    ")
    lines: list[str] = []
    blank_run = 0
    for line in text.split("\n"):
        if not line.strip():
            blank_run += 1
            if blank_run == 2:
                lines.append(_PARA_BLOCK)
            continue
        blank_run = 0
        lines.append(line.rstrip())
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _char_cells(ch: str) -> int:
    return 2 if glyphs.glyph_width(ch) > 8 else 1


def _line_cells(line: str) -> int:
    return sum(_char_cells(c) for c in line)


def _wrap(text: str, cols: int) -> list[str]:
    rows: list[str] = []
    for logical in text.split("\n"):
        if _line_cells(logical) <= cols:
            rows.append(logical)
            continue
        # Greedy word wrap with hard-break fallback for overlong tokens.
        cur = ""
        cur_cells = 0
        for token in logical.split(" "):
            token_cells = _line_cells(token)
            sep_cells = _line_cells(" ") if cur else 0
            if cur and cur_cells + sep_cells + token_cells <= cols:
                cur += " " + token
                cur_cells += sep_cells + token_cells
                continue
            if cur:
                rows.append(cur)
                cur, cur_cells = "", 0
            if token_cells <= cols:
                cur, cur_cells = token, token_cells
                continue
            # Hard-break the token cell by cell.
            piece = ""
            piece_cells = 0
            for ch in token:
                ch_cells = _char_cells(ch)
                if piece_cells + ch_cells > cols:
                    rows.append(piece)
                    piece, piece_cells = "", 0
                piece += ch
                piece_cells += ch_cells
            cur, cur_cells = piece, piece_cells
        if cur:
            rows.append(cur)
    return rows


def _blit_glyph(buf: bytearray, stride: int, x: int, y: int, gbytes: bytes, gw: int) -> None:
    """OR one glyph's packed rows into the frame buffer at pixel (x, y)."""
    row_bytes = gw // 8
    for r in range(GLYPH_H):
        base = (y + r) * stride
        for bi in range(row_bytes):
            v = gbytes[r * row_bytes + bi]
            if not v:
                continue
            px = x + bi * 8
            byte_i = base + px // 8
            sh = px % 8
            if sh == 0:
                buf[byte_i] |= v
            else:
                buf[byte_i] |= v >> sh
                buf[byte_i + 1] |= (v << (8 - sh)) & 0xFF


def render_page(rows: list[str], shape: Shape) -> bytes:
    """Rasterize one page of text rows into a 1-bit PNG (black ink on white)."""
    from PIL import Image

    width = shape.frame_w
    height = max(1, len(rows)) * shape.leading
    stride = width // 8
    buf = bytearray(stride * height)
    x = 0
    y = 0
    for line in rows:
        x = 0
        for ch in line:
            gw, gbytes = glyphs.glyph(ch)
            if x + gw > width:
                break  # clip: layout should never exceed, but never corrupt memory
            _blit_glyph(buf, stride, x, y, gbytes, gw)
            x += 16 if gw > 8 else shape.advance
        y += shape.leading
    img = Image.frombytes("1", (width, height), bytes(buf))
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def plan_frames(text: str, shape: Shape, max_frames: int) -> tuple[int, int]:
    """Count the frames ``text`` would occupy for ``shape`` without rendering.

    Returns ``(frame_count, omitted_oldest_pages)`` — the cheap economics probe
    the engine runs before committing to the vision path.
    """
    rows = _wrap(normalize(text), shape.cols)
    rpf = max(1, shape.rows_per_frame)
    n_pages = (len(rows) + rpf - 1) // rpf
    if max_frames > 0 and n_pages > max_frames:
        return max_frames, n_pages - max_frames
    return n_pages, 0


def render_frames(
    text: str, shape: Shape, max_frames: int
) -> tuple[list[bytes], int]:
    """Lay out ``text`` for ``shape`` and render at most ``max_frames`` PNGs.

    Returns ``(png_bytes_oldest_to_newest, omitted_oldest_pages)``. When the text
    exceeds the cap the OLDEST pages are dropped (the newest survive), and the
    count is returned so the summary can say so honestly.
    """
    rows = _wrap(normalize(text), shape.cols)
    rpf = max(1, shape.rows_per_frame)
    pages = [rows[i:i + rpf] for i in range(0, len(rows), rpf)]
    omitted = 0
    if len(pages) > max_frames > 0:
        omitted = len(pages) - max_frames
        pages = pages[-max_frames:]
    return [render_page(p, shape) for p in pages], omitted
