"""Per-model frame shapes for snapcompact.

Mirrors omp's snapcompact shape table: glyph advance (px per char), leading (px per
line), frame width, and an image-token billing model per provider family. Glyphs are
8x16 Unifont bitmaps (8 px wide, 16 px tall); wide (CJK etc.) glyphs span two cells.

Frame height "hugs the rows actually printed": a frame is exactly ``rows * leading``
pixels tall, never padded to a square.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GLYPH_W = 8
GLYPH_H = 16


@dataclass(frozen=True)
class Shape:
    name: str
    advance: int          # px per narrow char
    leading: int          # px per line
    frame_w: int          # frame width in px
    token_model: str      # billing model key

    @property
    def cols(self) -> int:
        return self.frame_w // self.advance

    @property
    def rows_per_frame(self) -> int:
        # Frame height never exceeds its width (square-ish cap, like the reference).
        return self.frame_w // self.leading


def _tokens_anthropic(w: int, h: int) -> int:
    return min((w * h) // 750, 4_784)


def _tokens_google(w: int, h: int) -> int:  # noqa: ARG001 - fixed per-image budget
    return 1_120


def _tokens_openai(w: int, h: int) -> int:
    return 85 + 170 * math.ceil(w / 512) * math.ceil(h / 512)


def _tokens_patch28(w: int, h: int) -> int:
    # 28px vision patches with 2x2 token merge (GLM-V style); documented estimate.
    return max(1, (w // 28) * (h // 28) // 4)


_TOKEN_MODELS = {
    "anthropic": _tokens_anthropic,
    "google": _tokens_google,
    "openai": _tokens_openai,
    "patch28": _tokens_patch28,
    "generic": _tokens_anthropic,
}


def frame_tokens(shape: Shape, rows: int) -> int:
    """Estimated billed tokens for a frame holding ``rows`` text lines."""
    w = shape.frame_w
    h = max(rows, 1) * shape.leading
    return _TOKEN_MODELS.get(shape.token_model, _tokens_anthropic)(w, h)


SHAPES: dict[str, Shape] = {
    # Anthropic: X.org-style 8-wide glyphs on an 11px advance, 16px leading.
    "claude": Shape("claude", 11, 16, 1_568, "anthropic"),
    # High-res Claude lines: wider frames under the 4,784 visual-token cap.
    "claude-hires": Shape("claude-hires", 11, 16, 1_932, "anthropic"),
    # Gemini: extra leading (22px), 2048px frames, fixed per-image budget.
    "gemini": Shape("gemini", 8, 22, 2_048, "google"),
    # GPT / Codex and the OpenAI-compatible default: 8on22 at 1568px.
    "openai": Shape("openai", 8, 22, 1_568, "openai"),
    # GLM / Kimi: 8-wide glyphs on a 16px pitch at 1568px (patch billing).
    "glm": Shape("glm", 8, 16, 1_568, "patch28"),
    # Qwen-VL and other 28px-patch local servers.
    "patch28": Shape("patch28", 8, 16, 1_568, "patch28"),
}

_HIRES_MARKERS = ("opus-4.7", "opus4.7", "opus-5", "opus5", "fable", "mythos")


def resolve_shape(model: str, forced: str = "auto") -> Shape:
    """Pick the eval-optimal shape for a model id (matches on the id, not the provider:
    a Claude routed through a gateway keeps the Claude shape)."""
    if forced and forced != "auto":
        if forced not in SHAPES:
            raise ValueError(f"unknown snapcompact shape '{forced}' (known: {sorted(SHAPES)})")
        return SHAPES[forced]
    m = (model or "").lower()
    if "claude" in m or "anthropic" in m:
        return SHAPES["claude-hires"] if any(t in m for t in _HIRES_MARKERS) else SHAPES["claude"]
    if "gemini" in m:
        return SHAPES["gemini"]
    if any(k in m for k in ("glm", "zhipu", "chatglm", "kimi", "moonshot")):
        return SHAPES["glm"]
    if any(k in m for k in ("qwen", "internvl", "minicpm")):
        return SHAPES["patch28"]
    if any(k in m for k in ("gpt", "codex", "o1", "o3", "o4")):
        return SHAPES["openai"]
    return SHAPES["openai"]
