type AgentEvent = {
  stream: string;
  data: Record<string, unknown>;
};

type ClaudeCliUsage = {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  total?: number;
};

type ClaudeCliOutput = {
  text: string;
  sessionId?: string;
  usage?: ClaudeCliUsage;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function readString(obj: Record<string, unknown>, key: string): string {
  const value = obj[key];
  return typeof value === "string" ? value : "";
}

function readNumber(obj: Record<string, unknown>, key: string): number | undefined {
  const value = obj[key];
  return typeof value === "number" ? value : undefined;
}

function parseUsage(raw: unknown): ClaudeCliUsage | undefined {
  if (!isRecord(raw)) return undefined;
  const input = readNumber(raw, "input_tokens") ?? readNumber(raw, "inputTokens");
  const output =
    readNumber(raw, "output_tokens") ?? readNumber(raw, "outputTokens");
  const cacheRead =
    readNumber(raw, "cache_read_input_tokens") ?? readNumber(raw, "cacheRead");
  const cacheWrite =
    readNumber(raw, "cache_creation_input_tokens") ??
    readNumber(raw, "cache_write_input_tokens") ??
    readNumber(raw, "cacheWrite");
  const total = readNumber(raw, "total_tokens") ?? readNumber(raw, "total");
  if (!input && !output && !cacheRead && !cacheWrite && !total) return undefined;
  return { input, output, cacheRead, cacheWrite, total };
}

function emitAgentEvent(
  params: {
    onAgentEvent?: (evt: AgentEvent) => void;
  },
  evt: AgentEvent,
) {
  params.onAgentEvent?.(evt);
}

function stripAnsi(raw: string): string {
  return raw.replace(/\u001b\[[0-9;]*m/g, "");
}

function tryParseJsonLine(rawLine: string): Record<string, unknown> | null {
  const line = rawLine.trim();
  if (!line) return null;
  try {
    const obj = JSON.parse(line);
    return isRecord(obj) ? obj : null;
  } catch {
    return null;
  }
}

function extractAssistantTextFromMessage(message: Record<string, unknown>): string {
  const content = message.content;
  if (!Array.isArray(content)) return "";
  let text = "";
  for (const part of content as unknown[]) {
    if (!isRecord(part)) continue;
    if (part.type === "text") {
      text += readString(part, "text");
    }
  }
  return text;
}

export function parseClaudeCliStream(params: {
  stdout: string;
  onAgentEvent?: (evt: AgentEvent) => void;
}): { output: ClaudeCliOutput; rawText: string } {
  const lines = params.stdout.split(/\r?\n/).filter(Boolean);

  let sessionId: string | undefined;
  let usage: ClaudeCliUsage | undefined;

  let assistantBuffer = "";
  let lastEmittedAssistant = "";

  const toolNamesById = new Map<string, string>();
  const toolArgsById = new Map<string, Record<string, unknown>>();

  const emitAssistant = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || trimmed === lastEmittedAssistant) return;
    lastEmittedAssistant = trimmed;
    emitAgentEvent(params, { stream: "assistant", data: { text: trimmed } });
  };

  for (const rawLine of lines) {
    const obj = tryParseJsonLine(stripAnsi(rawLine));
    if (!obj) continue;

    const sid = typeof obj.session_id === "string" ? obj.session_id.trim() : "";
    if (sid) sessionId = sid;

    if (obj.type === "stream_event") {
      const evt = obj.event;
      if (!isRecord(evt)) continue;

      if (evt.type === "content_block_delta") {
        const delta = evt.delta;
        if (isRecord(delta) && delta.type === "text_delta") {
          const text = readString(delta, "text");
          if (text) {
            assistantBuffer += text;
            emitAssistant(assistantBuffer);
          }
        }
      }

      if (evt.type === "message_delta") {
        usage = parseUsage(evt.usage) ?? usage;
      }

      continue;
    }

    if (obj.type === "assistant") {
      const msg = obj.message;
      if (isRecord(msg)) {
        usage = parseUsage(msg.usage) ?? usage;
        const text = extractAssistantTextFromMessage(msg);
        if (text.trim()) {
          assistantBuffer = text;
          emitAssistant(assistantBuffer);
        }

        const content = msg.content;
        if (Array.isArray(content)) {
          for (const part of content as unknown[]) {
            if (!isRecord(part)) continue;
            if (part.type !== "tool_use") continue;
            const toolCallId = readString(part, "id");
            const name = readString(part, "name");
            const input = isRecord(part.input) ? part.input : {};
            if (!toolCallId || !name) continue;
            toolNamesById.set(toolCallId, name);
            toolArgsById.set(toolCallId, input);
            emitAgentEvent(params, {
              stream: "tool",
              data: { phase: "start", name, toolCallId, args: input },
            });
          }
        }
      }
      continue;
    }

    if (obj.type === "user") {
      const msg = obj.message;
      if (isRecord(msg) && Array.isArray(msg.content)) {
        for (const part of msg.content as unknown[]) {
          if (!isRecord(part) || part.type !== "tool_result") continue;
          const toolCallId = readString(part, "tool_use_id");
          if (!toolCallId) continue;
          const name = toolNamesById.get(toolCallId);
          const isError = Boolean(part.is_error);
          const result = part.content;
          emitAgentEvent(params, {
            stream: "tool",
            data: {
              phase: "result",
              name,
              toolCallId,
              isError,
              result,
            },
          });
        }
      }
      continue;
    }

    if (obj.type === "result" && obj.subtype === "success") {
      const resultText = readString(obj, "result");
      if (resultText.trim()) {
        assistantBuffer = resultText;
        emitAssistant(assistantBuffer);
      }
      usage = parseUsage(obj.usage) ?? usage;
    }
  }

  const rawText = assistantBuffer.trim();
  return {
    rawText,
    output: {
      text: rawText,
      sessionId,
      usage,
    },
  };
}
