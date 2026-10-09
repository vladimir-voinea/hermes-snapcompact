"""SnapCompact context engine for Hermes.

A drop-in equivalent of oh-my-pi's snapcompact strategy: instead of asking a
summarizer LLM to paraphrase the discarded middle of a conversation, the middle is
serialized to bounded dense text, rasterized onto model-shaped 1-bit PNG frames of
pixel-font glyphs, and handed back to the vision-capable model as image parts on a
compaction summary row. The pass is local, deterministic and LLM-free — so it is
also safe for overflow recovery. On models without vision input (or when frames
would not actually shrink the transcript) it falls back to a deterministic bounded
text archive.

Pluggable via Hermes' context-engine slot: ``context.engine: "snapcompact"``.
Everything else (trigger thresholds, head/tail protection, tool-pair sanitization,
durable in-place commit, anti-growth guards, alternation) is inherited from the
built-in ``ContextCompressor`` this class subclasses; only the summarization step
and summary-row decoration are replaced.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    _COMPACTION_TAIL_MARKER,
    _SUMMARY_END_MARKER,
)
from agent.model_metadata import estimate_messages_tokens_rough

from . import render as _render
from . import serialize as _serial
from .serialize import SNAPCOMPACT_ARCHIVE_KEY
from .shapes import Shape, resolve_shape

logger = logging.getLogger(__name__)

_FRAME_NOTE = (
    "[snapcompact archive] The images above are a bitmap archive of the earlier "
    "conversation that was compacted away: pages of monospaced pixel text, read "
    "top-left to bottom-right, oldest first. They are an exact record, not a "
    "summary — consult them before claiming anything from earlier in the session "
    "is unknown."
)

_ARCHIVE_END_MARKER = (
    "--- END OF SNAPCOMPACT ARCHIVE — the real conversation continues below ---"
)

_LEADIN = (
    "[SNAPCOMPACT CONTEXT RESTORE] The earlier part of this conversation was "
    "compacted. The images attached to this message are its archive: dense pages "
    "of pixel text read top-left to bottom-right, oldest first. They are exact, "
    "not paraphrased. Read them like text before answering questions about "
    "earlier work, decisions, file paths, or instructions."
)

_NONVISION_HEADER = (
    "[SNAPCOMPACT CONTEXT RESTORE] The earlier part of this conversation was "
    "compacted with no vision model active, so a bounded text archive follows "
    "(head and tail kept, middle elided).\n\n"
)

_DEFAULTS: Dict[str, Any] = {
    "max_frames": 48,
    "shape": "auto",
    "assume_vision": "auto",          # auto | true | false
    "archive_source_max_chars": 48_000,
    "nonvision_max_chars": 24_000,
    "tool_result_max_chars": 2_000,
    "tool_result_head_ratio": 0.6,
    "tool_arg_max_chars": 500,
    "tool_call_max_chars": 2_000,
    "min_middle_tokens": 1_200,       # never bother below this
    "min_shrink_ratio": 0.9,          # frames must shrink the middle by >= 10%
}


def _load_hermes_config() -> Dict[str, Any]:
    import yaml

    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    try:
        with open(os.path.join(home, "config.yaml"), "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:  # noqa: BLE001 - config is optional; defaults apply
        return {}


def _image_token_cost() -> int:
    """Provider image price the rest of Hermes bills images at (flat default)."""
    try:
        from agent.image_token_cost import current_image_token_cost

        return int(current_image_token_cost())
    except Exception:  # noqa: BLE001
        return 1_500


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class SnapCompactEngine(ContextCompressor):
    """snapcompact strategy on top of the built-in compression pipeline."""

    def __init__(self) -> None:
        cfg = _load_hermes_config()
        comp = cfg.get("compression") or {}
        sc_raw = cfg.get("snapcompact") or {}
        self.sc_cfg: Dict[str, Any] = dict(_DEFAULTS)
        self.sc_cfg.update({k: v for k, v in sc_raw.items() if k in _DEFAULTS})
        super().__init__(
            model="",  # the host calls update_model() with the real model right after
            threshold_percent=float(comp.get("threshold", 0.50)),
            protect_first_n=int(comp.get("protect_first_n", 3)),
            protect_last_n=int(comp.get("protect_last_n", 20)),
            summary_target_ratio=float(comp.get("target_ratio", 0.20)),
            quiet_mode=False,
            abort_on_summary_failure=bool(comp.get("abort_on_summary_failure", False)),
            model_thresholds=comp.get("model_thresholds") or {},
            threshold_tokens_cap=comp.get("threshold_tokens"),
            min_tail_user_messages=int(comp.get("min_tail_user_messages", 1)),
            tail_mode=str(comp.get("tail_mode", "lean")),
        )
        self._shape: Optional[Shape] = None
        self._vision: Optional[bool] = None
        # Per-pass archive stash: {id(transcript_list): (archive, pngs, omitted,
        # n_middle, transcript_ref)}. Written in _summarize_window, consumed in
        # _finalize_compressed (same synchronous call); keyed by object identity
        # so concurrent passes for different sessions never collide.
        self._pass_stash: Dict[int, tuple] = {}

    # -- identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "snapcompact"

    # -- host lifecycle ---------------------------------------------------

    def update_model(
        self, model: str, context_length: int, base_url: str = "", api_key: Any = "",
        provider: str = "", api_mode: str = "", max_tokens: int | None = None,
    ) -> None:
        super().update_model(model, context_length, base_url, api_key, provider, api_mode, max_tokens)
        # The host constructs us before the model is known; base update_model only
        # recomputes thresholds, so mirror the built-in's identity fields here.
        self.model, self.base_url, self.provider, self.api_mode = model, base_url, provider, api_mode
        self._shape = resolve_shape(model, str(self.sc_cfg.get("shape", "auto")))
        self._vision = None  # re-probe for the new model
        if not self.quiet_mode:
            logger.info(
                "snapcompact: model=%s shape=%s (%d cols x %d rows/frame)",
                model, self._shape.name, self._shape.cols, self._shape.rows_per_frame,
            )

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        shape = self._shape
        status["engine"] = self.name
        status["snapcompact"] = {
            "shape": shape.name if shape else None,
            "max_frames": self.sc_cfg.get("max_frames"),
            "vision_capable": None if self._vision is None else self._vision,
        }
        return status

    # -- vision -----------------------------------------------------------

    def _vision_capable(self) -> bool:
        mode = self.sc_cfg.get("assume_vision", "auto")
        if mode is True:
            return True
        if mode is False:
            return False
        if self._vision is None:
            resolved = None
            try:
                from agent.image_routing import _lookup_supports_vision

                resolved = _lookup_supports_vision(
                    self.provider or "", self.model or "", _load_hermes_config(),
                    requested_provider=self.provider or "",
                )
            except Exception:  # noqa: BLE001 - probe failure == unknown
                resolved = None
            self._vision = bool(resolved)
            if not self._vision:
                logger.warning(
                    "snapcompact: model %s has no confirmed image input; falling back to "
                    "deterministic bounded text archives", self.model or "?",
                )
        return self._vision

    def _shape_for(self) -> Shape:
        return self._shape or resolve_shape(self.model, str(self.sc_cfg.get("shape", "auto")))

    # -- compaction internals (replacing the LLM summarizer) ---------------

    def compress(
        self, messages: List[Dict[str, Any]], current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None, force: bool = False, memory_context: str = "",
        bypass_cooldown: bool = False,
    ) -> List[Dict[str, Any]]:
        # Cheap pre-gate: middles smaller than ~2 image tokens-worth can never
        # shrink into frames; skip without side effects.
        if len(messages) > 6 and not force:
            head = self._protect_head_size(messages)
            tail_cut = -self.protect_last_n if self.protect_last_n else None
            est = estimate_messages_tokens_rough(messages[head:tail_cut])
            floor = max(int(self.sc_cfg.get("min_middle_tokens", 1_200)), 2 * _image_token_cost())
            if est and est < floor:
                return messages
        return super().compress(
            messages, current_tokens=current_tokens, focus_topic=focus_topic, force=force,
            memory_context=memory_context, bypass_cooldown=bypass_cooldown,
        )

    def _summarize_window(
        self, messages: List[Dict[str, Any]], turns_to_summarize: List[Dict[str, Any]],
        scan: Any, focus_topic: Optional[str], memory_context: str, bypass_cooldown: bool,
    ) -> Optional[str]:
        """Deterministic summary text; the archive itself rides on the frames.

        No LLM call is made in either path, so compression cannot fail on a
        summary-model outage and stays safe for overflow recovery. The vision
        path is only taken when the frames would actually shrink the middle.
        """
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        archive = _serialize_middle(turns_to_summarize, self.sc_cfg)
        use_frames = False
        pngs: list = []
        omitted = 0
        if self._vision_capable() and archive.strip():
            shape = self._shape_for()
            n_frames, _omitted = _render.plan_frames(
                archive, shape, int(self.sc_cfg.get("max_frames", 48)),
            )
            est_middle = estimate_messages_tokens_rough(turns_to_summarize)
            est_frames = n_frames * _image_token_cost()
            use_frames = bool(
                n_frames
                and est_middle
                and est_frames < est_middle * float(self.sc_cfg.get("min_shrink_ratio", 0.9))
            )
            if not use_frames:
                logger.info(
                    "snapcompact: frames (%d, ~%d tok) would not shrink the middle "
                    "(~%d tok); using a bounded text archive instead",
                    n_frames, est_frames, est_middle,
                )
            else:
                pngs, omitted = _render.render_frames(
                    archive, shape, int(self.sc_cfg.get("max_frames", 48)),
                )
        # Stash this pass's archive for _finalize_compressed (same call stack).
        # Keyed by the transcript object's identity (and holding a reference so
        # the id cannot be recycled while cached); bounded to the last few passes.
        self._pass_stash[id(messages)] = (
            archive, pngs, omitted, len(turns_to_summarize), messages,
        )
        while len(self._pass_stash) > 4:
            oldest_key = next(iter(self._pass_stash))
            del self._pass_stash[oldest_key]
        if use_frames:
            sections = [_LEADIN]
        else:
            sections = [
                _NONVISION_HEADER
                + _serial.bound_archive_text(
                    archive, int(self.sc_cfg.get("nonvision_max_chars", 24_000)),
                )
            ]
        if memory_context.strip():
            sections.append("[PROVIDER CONTEXT]\n" + memory_context.strip())
        if focus_topic:
            sections.append(f"[USER FOCUS for this compaction: {focus_topic}]")
        return "\n\n".join(sections)

    def _finalize_compressed(
        self, compressed: List[Dict[str, Any]], messages: List[Dict[str, Any]], n_messages: int,
    ) -> List[Dict[str, Any]]:
        compressed = super()._finalize_compressed(compressed, messages, n_messages)
        try:
            # Runs after all base cleanup (tool pairs, media stripping), so the
            # frames cannot be touched by it. Fail-open: a text-only summary remains.
            self._attach_frames(compressed, messages)
        except Exception:  # noqa: BLE001
            logger.warning("snapcompact: frame attachment failed; keeping text summary", exc_info=True)
        return compressed

    # -- frame attachment ---------------------------------------------------

    def _attach_frames(self, compressed: List[Dict[str, Any]], messages: List[Dict[str, Any]]) -> None:
        if not compressed or not self._vision_capable():
            return
        stash = self._pass_stash.pop(id(messages), None)
        if stash is None:
            # No stash for this transcript (unexpected path): keep the text
            # summary rather than risk attaching an archive for the wrong window.
            logger.warning("snapcompact: no archive stash for this pass; skipping frames")
            return
        archive, pngs, omitted, n_middle, _ref = stash
        shape = self._shape_for()

        idx = next(
            (i for i, m in enumerate(compressed)
             if isinstance(m, dict) and m.get(COMPRESSED_SUMMARY_METADATA_KEY)),
            None,
        )
        if idx is None:
            logger.warning("snapcompact: no summary row found; frames not attached")
            self._pass_stash[id(messages)] = stash  # restore for possible retry
            return
        row = compressed[idx]
        role = row.get("role")
        # The archive source is durable regardless of frames: it is what later
        # compactions re-embed (serialize handles the summary-row shape), so it
        # must be set on every pass that produced an archive — not only when
        # frames won the economics.
        row[_serial.SNAPCOMPACT_ARCHIVE_KEY] = self._last_archive_meta = {
            "text": _serial.bound_archive_text(
                archive, int(self.sc_cfg.get("archive_source_max_chars", 48_000)),
            ),
            "frames": len(pngs), "omitted_oldest": omitted, "shape": shape.name,
            "model": self.model, "middle_messages": n_middle,
        }
        if not pngs:
            return  # economics said no; the summary row already carries the text archive
        image_parts = [{"type": "image_url", "image_url": {"url": _png_data_url(p)}} for p in pngs]
        if role == "user":
            # Frames belong inside the user-role summary row, inside its summary
            # block (carried rows carry the whole summary before their end marker).
            if row.get(_COMPACTION_TAIL_MARKER):
                _prepend_parts(row, [_FRAME_NOTE_PART] + image_parts)
            else:
                summary_text = str(row.get("content") or "")
                marker_at = summary_text.rfind(_SUMMARY_END_MARKER)
                head_text = summary_text[:marker_at].rstrip("\n") if marker_at >= 0 else summary_text
                row["content"] = (
                    [{"type": "text", "text": head_text}]
                    + image_parts
                    + [{"type": "text", "text": _SUMMARY_END_MARKER}]
                )
        else:
            # Assistant-role summary: frames go into a user-role slot after it so
            # image parts never ride in an assistant message.
            self._place_frames_row(compressed, idx, image_parts)
        logger.info(
            "snapcompact: attached %d frame(s) (%s) for %d middle messages%s",
            len(pngs), shape.name, n_middle,
            f", {omitted} oldest page(s) omitted" if omitted else "",
        )

    def _place_frames_row(
        self, compressed: List[Dict[str, Any]], summary_idx: int,
        image_parts: List[Dict[str, Any]],
    ) -> None:
        """Find an alternation-safe user-role slot for frames after a non-user
        summary row. Never splits an assistant->tool pair; when the only slot is a
        real user row, frames are prepended with an explicit archive-end marker so
        the user's own content stays clearly delimited (no summary metadata is set
        on real rows, so display never hides them)."""
        i = summary_idx + 1
        n = len(compressed)
        while i < n and isinstance(compressed[i], dict) and compressed[i].get("role") == "tool":
            i += 1
        if i < n and isinstance(compressed[i], dict) and compressed[i].get("role") == "user":
            _prepend_parts(compressed[i], [_FRAME_NOTE_PART] + image_parts + [
                {"type": "text", "text": _ARCHIVE_END_MARKER},
            ])
            # The archive rides with the frames: this row becomes the durable
            # carrier later compactions re-embed from (metadata keys are the
            # reliable signal here; this row keeps the user's real turn after
            # the end marker, so it must not get summary-row display metadata).
            compressed[i][SNAPCOMPACT_ARCHIVE_KEY] = dict(self._last_archive_meta)
            return
        compressed.insert(i, {
            "role": "user",
            "content": [_FRAME_NOTE_PART] + image_parts,
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        })


_FRAME_NOTE_PART: Dict[str, Any] = {"type": "text", "text": _FRAME_NOTE}


def _prepend_parts(row: Dict[str, Any], parts: List[Dict[str, Any]]) -> None:
    content = row.get("content")
    if isinstance(content, list):
        row["content"] = parts + list(content)
    else:
        row["content"] = parts + [{"type": "text", "text": str(content or "")}]
    from agent.turn_context import drop_stale_api_content

    drop_stale_api_content(row)


def _serialize_middle(middle: List[Dict[str, Any]], cfg: Dict[str, Any]) -> str:
    return _serial.serialize_messages(
        middle,
        tool_result_max_chars=int(cfg.get("tool_result_max_chars", 2_000)),
        tool_result_head_ratio=float(cfg.get("tool_result_head_ratio", 0.6)),
        tool_arg_max_chars=int(cfg.get("tool_arg_max_chars", 500)),
        tool_call_max_chars=int(cfg.get("tool_call_max_chars", 2_000)),
    )
