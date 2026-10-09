# hermes-snapcompact

**Bitmap-frame context compression for [Hermes Agent](https://github.com/NousResearch/hermes-agent) — a drop-in port of oh-my-pi's `snapcompact` strategy.**

When a conversation outgrows its context window, agents normally ask a summarizer LLM to *paraphrase* the discarded middle. snapcompact instead **prints that history onto dense PNG frames of pixel-font glyphs** and hands the images back to the vision-capable model as an exact, lossless-format record — no summarizer call, no paraphrase drift, no API cost for the compression pass itself.

Same idea as [oh-my-pi](https://github.com/can1357/oh-my-pi)'s `@oh-my-pi/snapcompact`: the compression pass is local, deterministic rasterization; the model *reads its own history like an OCR test* instead of trusting a lossy summary of it.

## How it works

```
discarded middle ──▶ serialize (bounded dense text) ──▶ word-wrap to shape ──▶ paginate
      ──▶ rasterize 1-bit PNG frames (8x16 Unifont glyphs) ──▶ attach as image parts
             on the compaction summary row ──▶ vision model reads them back
```

1. **Serialize** — the compressible middle becomes conversation-dense text: `[USER]` / `[ASSISTANT]` / `[TOOL name]` / `[TOOL_CALL name(args)]` blocks. Tool results are truncated head+tail (default 2,000 chars at a 0.6 head ratio); tool-call arguments are capped per value (500) and per call (2,000). Prior compaction archives are re-embedded from their bounded source text, so chained compactions never OCR their own PNGs.
2. **Shape** — the frame geometry (glyph advance, leading, frame width) resolves from the **model id**, matching omp's per-family tuning: Claude, Gemini, GPT/Codex, GLM/Kimi, and a generic 28px-patch shape (Qwen-VL & friends). A Claude routed through a gateway keeps the Claude shape.
3. **Render** — rows of text are rasterized into 1-bit black-on-white PNGs whose height hugs the rows actually printed. Missing glyphs fall back `?` → space; CJK renders full-width. Same text + shape ⇒ same bytes.
4. **Attach** — frames ride inside the alternation-safe summary block (never inside an assistant message, never splitting a tool pair). A short lead-in tells the model the images are an *exact* archive, not a summary. A bounded source text rides along in row metadata as the durable archive for later re-compaction.
5. **Fallbacks, by design:**
   - Model has no image input → deterministic bounded **text archive** (head/tail kept, middle elided) instead of frames.
   - Frames wouldn't actually shrink the middle (billing the images at Hermes' per-image token cost) → text archive. The economics gate is `n_frames * image_token_cost < middle_tokens * min_shrink_ratio` (default 0.9).
   - Any failure → fail-open to the ordinary text summary; compression never blocks the session.

Everything else — trigger thresholds, head/tail protection, tool-pair sanitization, durable in-place commits, anti-growth guards, role alternation — is **inherited from Hermes' built-in `ContextCompressor`**; only the summarization step and summary-row decoration are replaced. The pass is LLM-free, so it is also safe for overflow recovery.

## Install

Requires a Hermes Agent checkout with its venv and Pillow (already a Hermes dependency).

```bash
# 1. Copy the plugin into Hermes' context-engine directory
cp -R snapcompact ~/.hermes/hermes-agent/plugins/context_engine/

# 2. Select it
hermes config set context.engine snapcompact
```

That's it. New sessions announce `Using context engine: snapcompact`; `/compress` works as usual.

## Configuration

`compression.*` keeps governing trigger thresholds/protection (the engine is a `ContextCompressor` subclass). The vision strategy itself reads a top-level `snapcompact:` block:

```yaml
snapcompact:
  max_frames: 48               # hard cap on frames per compaction (oldest pages elided)
  shape: auto                  # auto | claude | claude-hires | gemini | openai | glm | patch28
  assume_vision: auto          # auto (probe) | true | false — force the frames path
  min_shrink_ratio: 0.9        # frames must shrink the middle by >= 10% or fall back
  min_middle_tokens: 1200      # skip passes that cannot possibly pay for themselves
  archive_source_max_chars: 48000
  nonvision_max_chars: 24000
  tool_result_max_chars: 2000
  tool_result_head_ratio: 0.6
  tool_arg_max_chars: 500
  tool_call_max_chars: 2000
```

Per-model compaction thresholds keep working via `compression.model_thresholds`.

## Does the model actually read the frames?

Yes. Verified end-to-end against a GLM-5.3-Flash-EXL3 server (`vLLM`-style OpenAI-compatible endpoint): a single frame containing fabricated "archived" facts was sent as the only source, and the model answered all four questions exactly:

> **Q:** project name / staging cluster / release captain / maintenance passphrase?
> **A:** Project Cobalt Falcon / blue-heron-7 / Dana Whitfield / TUNGSTEN-OTTER-42 ✓

## Tests

The venv deliberately omits pytest, so the suite ships with a tiny runner:

```bash
cd ~/.hermes/hermes-agent
./venv/bin/python plugins/context_engine/snapcompact/tests/run_tests.py
```

Covers: shape resolution per model family, deterministic rendering, frame geometry/caps, serializer caps and merged-carrier recovery, alternation-safe attachment, transcript shrinkage, non-vision fallback, and chained re-compaction re-embedding prior archives from their source text.

## The font

`fonts/unifont16.bin` is a compiled glyph cache extracted from **GNU Unifont** (`unifont-17.0.04` BDF, 8×16 pixel glyphs; GPL v2 **with the GNU Font Embedding Exception** — see `NOTICE-Unifont`). The cache holds unaltered bitmaps; `scripts/extract_unifont.py` rebuilds it from upstream so nothing is opaque:

```bash
./scripts/extract_unifont.py unifont-17.0.04.bdf.gz > new-unifont16.bin
```

## Status

- **13/13 engine tests pass** on Hermes 0.21.3.
- One engine active at a time (Hermes' context-engine slot); `context.engine: snapcompact` selects it.
- Not affiliated with oh-my-pi or Hermes Agent; this is an independent port.

## License

MIT for the code. GNU Unifont glyph data is © its authors under GPLv2 *with the font embedding exception* — see `NOTICE-Unifont`.
