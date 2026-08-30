# Pi integration

Merge the `qwen38-r9v` provider from `integrations/pi/models.example.json` into `~/.pi/agent/models.json`, then copy `integrations/pi/qwen38-performance.ts` to `~/.pi/agent/extensions/`.

Launch the interactive coding agent:

```bash
pi --model qwen38-r9v/qwen3.8-flash-next
```

Pass an image through the OpenAI-compatible vision route:

```bash
pi --model qwen38-r9v/qwen3.8-flash-next @/path/to/image.png \
  "Describe this image"
```

The performance extension shows one client-observed sample for every model call:

```text
Qwen call 1 • in 466 • out 63 • TTFT 2.798s • PP≈166.5 tok/s • TG≈58.35 tok/s
```

- Input and output token counts are reported by vLLM's final streaming usage event.
- TTFT is measured from Pi's turn start to its first streamed text, thinking, or tool-call delta.
- `PP≈` is prompt tokens divided by TTFT.
- `TG≈` is all tokens after the first divided by post-first-token wall time.
- These are client-observed rates, so TTFT includes serialization, queueing, and API overhead.
- Prefix-cache tokens are included in the prompt total and displayed separately.
- Tool workflows report each model call independently.

Run `/qwen-perf` to show the last sample again. If Pi was open when the extension was installed, run `/reload`.
