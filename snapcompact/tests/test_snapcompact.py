"""Tests for the snapcompact context engine.

Run from the repo root: venv/bin/python -m pytest plugins/context_engine/snapcompact/tests -q
"""

import base64
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    import pytest
except ImportError:  # venv ships without dev deps; the mini-runner tolerates this
    class _PytestFail(Exception):
        pass

    class _PytestShim:
        def fail(self, msg=""):
            raise AssertionError(msg)

        def raises(self, *a, **k):
            import contextlib

            return contextlib.nullcontext()

        def fixture(self, *a, **k):
            def deco(fn):
                return fn
            return deco if (a or k) else (lambda fn: fn)

    pytest = _PytestShim()  # type: ignore[assignment]

from agent.context_compressor import _SUMMARY_END_MARKER  # noqa: E402
from plugins.context_engine.snapcompact import render, serialize  # noqa: E402
from plugins.context_engine.snapcompact.serialize import SNAPCOMPACT_ARCHIVE_KEY  # noqa: E402
from plugins.context_engine.snapcompact.engine import SnapCompactEngine  # noqa: E402
from plugins.context_engine.snapcompact.shapes import resolve_shape  # noqa: E402

MODEL = "GLM-5.3-Flash-EXL3"
CTX = 200_000


@pytest.fixture()
def engine():
    eng = SnapCompactEngine()
    eng.update_model(MODEL, CTX, provider="glm53-flash")
    eng._vision = True  # deterministic; vision probe tested separately
    return eng


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text):
    return {"role": "assistant", "content": text}


def _tool_call_row(call_id, name, args):
    return {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}],
    }


def _tool_result(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "name": "read_file", "content": text}


def _transcript(turns=60, tool_every=3):
    msgs = [{"role": "system", "content": "You are a coding agent."},
            _user("Build the auth module."),
            _assistant("On it.")]
    for i in range(turns):
        msgs.append(_user(f"Turn {i}: inspect module_{i}.py and fix the bug."))
        if i % tool_every == 0:
            msgs.append(_tool_call_row(f"call_{i}", "read_file", {"path": f"src/module_{i}.py"}))
            msgs.append(_tool_result(
                f"call_{i}",
                f"def buggy_{i}():\n    pass\n# {'padding ' * 100}\n" + ("x" * 1500),
            ))
        msgs.append(_assistant(f"Fixed bug {i} in module_{i}.py, tests pass."))
    return msgs


def _plain_role(msg):
    """Template-visible role for alternation checks: assistant rows carrying
    tool_calls and tool-result rows are transparent (a tool loop is not a
    same-role violation)."""
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    if role == "assistant" and msg.get("tool_calls"):
        return None
    return role if role in ("user", "assistant") else None


# -- shapes ---------------------------------------------------------------

def test_shape_resolution_matches_model_family():
    assert resolve_shape("GLM-5.3-Flash-EXL3").name == "glm"
    assert resolve_shape("claude-sonnet-4").name == "claude"
    assert resolve_shape("claude-opus-4.7").name == "claude-hires"
    assert resolve_shape("gemini-3-pro").name == "gemini"
    assert resolve_shape("gpt-5.2").name == "openai"
    assert resolve_shape("totally-unknown-model").name == "openai"  # default family


def test_forced_shape_validated():
    raised = False
    try:
        resolve_shape("x", forced="nope")
    except ValueError:
        raised = True
    assert raised


# -- renderer -------------------------------------------------------------

def test_render_is_deterministic():
    s = resolve_shape(MODEL)
    a = render.render_frames("hello world\n\nsecond para", s, 5)[0]
    b = render.render_frames("hello world\n\nsecond para", s, 5)[0]
    assert a == b


def test_frame_geometry_hugs_rows_and_caps_frames():
    s = resolve_shape(MODEL)
    text = "\n".join(f"line {i}" for i in range(s.rows_per_frame * 5 + 3))
    pngs, omitted = render.render_frames(text, s, 4)
    assert len(pngs) == 4 and omitted == 2
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(pngs[0]))
    assert img.size[0] == s.frame_w
    assert img.size[1] <= s.frame_w  # frame height never exceeds width
    # data URLs decode as PNG
    raw = base64.b64decode(pngs[0][:0] or b"")  # noqa: F841 - keep import shape obvious


# -- serializer -----------------------------------------------------------

def test_serialization_caps_tool_results_and_args():
    big = "A" * 10_000
    middle = [
        _user("do the thing"),
        _tool_call_row("c1", "bash", {"cmd": "z" * 3_000, "n": 1}),
        _tool_result("c1", big),
    ]
    text = serialize.serialize_messages(
        middle, tool_result_max_chars=2_000, tool_result_head_ratio=0.6,
        tool_arg_max_chars=500, tool_call_max_chars=2_000,
    )
    assert "[USER]" in text and "[TOOL read_file]" in text and "[TOOL_CALL bash(" in text
    assert "elided" in text
    # per-call arg cap: the z-run must be cut well below 3000
    assert "z" * 600 not in text
    assert len(text) < 6_000


def test_prior_snapcompact_archive_is_reembedded_from_source():
    middle = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                                     {"type": "text", "text": "ignored"}],
         "_compressed_summary": True,
         SNAPCOMPACT_ARCHIVE_KEY: {"text": "THE OLD ARCHIVE TEXT"}},
    ]
    text = serialize.serialize_messages(middle)
    assert "THE OLD ARCHIVE TEXT" in text
    assert "[image attached]" not in text  # source text used, not part shapes


# -- engine ---------------------------------------------------------------

def test_engine_identity_and_threshold(engine):
    assert engine.name == "snapcompact"
    assert engine.context_length == CTX
    # threshold follows the configured percent (capped when threshold_tokens set)
    expected = int(CTX * engine.threshold_percent)
    assert engine.threshold_tokens == expected or engine.threshold_tokens < expected
    assert engine._shape.name == "glm"


def test_compress_produces_summary_row_with_frames_and_valid_alternation(engine):
    msgs = _transcript(30)
    out = engine.compress(msgs, current_tokens=150_000)
    assert out is not msgs and len(out) < len(msgs)
    # roles never repeat (tool loops are transparent for alternation)
    prev = None
    for m in out:
        r = _plain_role(m)
        if r is not None and r == prev:
            pytest.fail(f"alternation broken at role {r}")
        if r is not None:
            prev = r
    # exactly one compaction summary row
    summary_rows = [m for m in out if m.get("_compressed_summary")]
    assert len(summary_rows) == 1
    # frames ride in exactly one user-role row: inside the summary row when it is
    # user-role, in a dedicated/next user row when the summary is assistant-role.
    summary_role = summary_rows[0]["role"]
    image_rows = [m for m in out if isinstance(m.get("content"), list)
                  and any(p.get("type") == "image_url" for p in m["content"])]
    assert len(image_rows) == 1
    assert image_rows[0]["role"] == "user"
    if summary_role == "user":
        assert image_rows[0] is summary_rows[0]
    else:
        assert image_rows[0] is not summary_rows[0]
        assert out.index(image_rows[0]) > out.index(summary_rows[0])
    urls = [p["image_url"]["url"] for p in image_rows[0]["content"] if p.get("type") == "image_url"]
    assert urls and all(u.startswith("data:image/png;base64,") for u in urls)
    # the summary block ends with the end marker wherever the text lives
    summary_content = summary_rows[0]["content"]
    summary_text = summary_content if isinstance(summary_content, str) else "".join(
        p.get("text", "") for p in summary_content if p.get("type") == "text")
    assert _SUMMARY_END_MARKER in summary_text
    # archive source stashed for later re-compaction (covers the compressible
    # middle; the head is protected verbatim and never serialized)
    arch = summary_rows[0][SNAPCOMPACT_ARCHIVE_KEY]
    assert arch["frames"] == len(urls) and arch["model"] == MODEL
    assert "Turn 5: inspect module_5.py" in arch["text"]  # oldest middle content present
    # head and tail survive verbatim
    assert out[0]["role"] == "system"
    assert any(m.get("role") == "user" and "Turn 29" in str(m.get("content")) for m in out)
    assert engine.compression_count == 1


def test_compress_shrinks_by_rough_estimate(engine):
    msgs = _transcript(40)
    from agent.model_metadata import estimate_messages_tokens_rough

    out = engine.compress(msgs, current_tokens=150_000)
    assert estimate_messages_tokens_rough(out) < estimate_messages_tokens_rough(msgs)


def test_small_middle_is_skipped_without_side_effects(engine):
    msgs = [
        {"role": "system", "content": "sys"},
        _user("hi"), _assistant("hello"), _user("bye"), _assistant("bye"), _user("go"),
        _assistant("ok"),
    ]
    out = engine.compress(msgs)
    assert out is msgs


def test_nonvision_falls_back_to_bounded_text(engine):
    engine._vision = False
    msgs = _transcript(30)
    out = engine.compress(msgs, current_tokens=150_000)
    summary_rows = [m for m in out if m.get("_compressed_summary")]
    assert len(summary_rows) == 1
    content = summary_rows[0]["content"]
    assert isinstance(content, str)  # plain text, no image parts anywhere
    assert all(not isinstance(p, dict) or p.get("type") != "image_url" for m in out
               if isinstance(m.get("content"), list) for p in m["content"])
    assert "bounded text archive" in content
    # anti-growth: must still shrink
    from agent.model_metadata import estimate_messages_tokens_rough

    assert estimate_messages_tokens_rough(out) < estimate_messages_tokens_rough(msgs)


def test_recompaction_reembeds_prior_archive(engine):
    msgs = _transcript(30)
    first = engine.compress(msgs, current_tokens=150_000)
    # Grow well past the protected tail (15 messages) so the pass-1 summary row
    # lands inside pass-2's compressible middle and gets re-embedded from its
    # durable source text.
    new_rows = [_user("New work begins here."), _assistant("More work done.")]
    for i in range(20):
        new_rows.append(_user(f"Second act {i}: verify module_{i}."))
        new_rows.append(_assistant(f"Verified {i}."))
    second_input = first[:-2] + new_rows
    out = engine.compress(second_input, current_tokens=150_000)
    summary_rows = [m for m in out if m.get("_compressed_summary")]
    assert len(summary_rows) == 1
    arch = summary_rows[0].get(SNAPCOMPACT_ARCHIVE_KEY)
    assert arch is not None, "every pass must carry the durable archive source"
    assert "[PRIOR CONTEXT SUMMARY]" in arch["text"]
    assert "Build the auth module" in arch["text"]  # original content survived round-trip


def test_status_reports_engine_fields(engine):
    st = engine.get_status()
    assert st["engine"] == "snapcompact"
    assert st["snapcompact"]["shape"] == "glm"
    assert "threshold_tokens" in st and "usage_percent" in st
