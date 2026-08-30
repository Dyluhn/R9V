import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PROVIDERS = new Set(["qwen38-r9v", "qwen38-radiance"]);
const WIDGET_KEY = "qwen38-performance";

interface ActiveCall {
  turnIndex: number;
  startedMs: number;
  firstOutputMs?: number;
}

interface PerfSample {
  turnIndex: number;
  promptTokens: number;
  cachedTokens: number;
  outputTokens: number;
  ttftSeconds: number;
  decodeSeconds: number;
  pp?: number;
  tg?: number;
}

function formatRate(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "n/a";
  if (value >= 1000) return value.toFixed(0);
  if (value >= 100) return value.toFixed(1);
  return value.toFixed(2);
}

function formatSample(sample: PerfSample): string {
  const cache = sample.cachedTokens > 0 ? `, cached ${sample.cachedTokens}` : "";
  return [
    `Qwen call ${sample.turnIndex + 1}`,
    `in ${sample.promptTokens}${cache}`,
    `out ${sample.outputTokens}`,
    `TTFT ${sample.ttftSeconds.toFixed(3)}s`,
    `PP≈${formatRate(sample.pp)} tok/s`,
    `TG≈${formatRate(sample.tg)} tok/s`,
  ].join("  •  ");
}

export default function (pi: ExtensionAPI) {
  let active: ActiveCall | undefined;
  let last: PerfSample | undefined;

  pi.on("turn_start", async (event) => {
    active = { turnIndex: event.turnIndex, startedMs: performance.now() };
  });

  pi.on("message_update", async (event) => {
    if (!active || event.message.role !== "assistant") return;
    if (!PROVIDERS.has(event.message.provider)) return;
    if (active.firstOutputMs !== undefined) return;

    const update = event.assistantMessageEvent;
    const hasOutput =
      (update.type === "text_delta" && update.delta.length > 0) ||
      (update.type === "thinking_delta" && update.delta.length > 0) ||
      (update.type === "toolcall_delta" && update.delta.length > 0);
    if (hasOutput) active.firstOutputMs = performance.now();
  });

  pi.on("turn_end", async (event, ctx) => {
    if (!active || event.message.role !== "assistant") return;
    if (!PROVIDERS.has(event.message.provider)) return;

    const endedMs = performance.now();
    const firstMs = active.firstOutputMs ?? endedMs;
    const usage = event.message.usage;
    const cachedTokens = usage.cacheRead ?? 0;
    const promptTokens = (usage.input ?? 0) + cachedTokens;
    const outputTokens = usage.output ?? 0;
    const ttftSeconds = Math.max(0, firstMs - active.startedMs) / 1000;
    const decodeSeconds = Math.max(0, endedMs - firstMs) / 1000;

    last = {
      turnIndex: active.turnIndex,
      promptTokens,
      cachedTokens,
      outputTokens,
      ttftSeconds,
      decodeSeconds,
      pp: promptTokens > 0 && ttftSeconds > 0 ? promptTokens / ttftSeconds : undefined,
      tg:
        outputTokens > 1 && decodeSeconds > 0
          ? (outputTokens - 1) / decodeSeconds
          : undefined,
    };

    const line = formatSample(last);
    if (ctx.hasUI) {
      ctx.ui.setWidget(WIDGET_KEY, [line], { placement: "aboveEditor" });
      ctx.ui.setStatus(
        WIDGET_KEY,
        `PP ${formatRate(last.pp)} | TG ${formatRate(last.tg)} tok/s`,
      );
    } else {
      process.stderr.write(`\n[${line}]\n`);
    }
    active = undefined;
  });

  pi.registerCommand("qwen-perf", {
    description: "Show performance for the last local Qwen model call",
    handler: async (_args, ctx) => {
      ctx.ui.notify(last ? formatSample(last) : "No Qwen performance sample yet.", "info");
    },
  });

  pi.on("session_shutdown", async (_event, ctx) => {
    if (!ctx.hasUI) return;
    ctx.ui.setWidget(WIDGET_KEY, undefined);
    ctx.ui.setStatus(WIDGET_KEY, undefined);
  });
}
