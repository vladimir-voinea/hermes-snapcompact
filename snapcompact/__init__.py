"""snapcompact: bitmap-frame context compression engine for Hermes.

Discarded conversation history is serialized to bounded dense text, rasterized
onto model-shaped PNG frames of 8x16 pixel glyphs, and re-attached as image
parts on the compaction summary row. Deterministic, local, LLM-free; falls back
to a bounded text archive on non-vision models. Activate with
``context.engine: "snapcompact"`` in config.yaml.
"""

from plugins.context_engine.snapcompact.engine import SnapCompactEngine

__all__ = ["SnapCompactEngine"]
