import type { AgentTool } from "@mariozechner/pi-agent-core";

type ClaudeCliStreamLine = Record<string, unknown>;

type ToolUse = {
  id: string;
  name: string;
  input: Record<string, unknown>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function readString(obj: Record<string, unknown>, key: string): string {
  const value = obj[key];
  return typeof value === "string" ? value : "";
}

function stripAnsi(raw: string): string {
  return raw.replace(/\u001b\[[0-9;]*m/g, "");
}

export function buildToolAllowlist(tools: AgentTool[]): string {
  const names = tools
    .map((tool) => tool.name)
    .filter((name): name is string => typeof name === "string" && !!name.trim());
  // Claude Code expects comma-separated tool names.
  return names.join(",");
}

export function buildStreamJsonUserMessage(prompt: string): string {
  return JSON.stringify({
    type: "user",
    message: {
      role: "user",
      content: [
        {
          type: "text",
          text: prompt,
        },
      ],
    },
  });
}

export function buildStreamJsonToolResult(params: {
  toolUseId: string;
  content: string;
  isError: boolean;
}): string {
  return JSON.stringify({
    type: "user",
    message: {
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: params.toolUseId,
          content: params.content,
          is_error: params.isError,
        },
      ],
    },
  });
}

export function parseStreamJsonLine(rawLine: string): ClaudeCliStreamLine | null {
  const line = stripAnsi(rawLine).trim();
  if (!line) return null;
  try {
    const obj = JSON.parse(line);
    return isRecord(obj) ? obj : null;
  } catch {
    return null;
  }
}

export function extractToolUsesFromAssistantLine(
  obj: ClaudeCliStreamLine,
): ToolUse[] {
  if (obj.type !== "assistant") return [];
  const message = obj.message;
  if (!isRecord(message)) return [];
  const content = message.content;
  if (!Array.isArray(content)) return [];
  const toolUses: ToolUse[] = [];
  for (const part of content as unknown[]) {
    if (!isRecord(part) || part.type !== "tool_use") continue;
    const id = readString(part, "id");
    const name = readString(part, "name");
    const input = isRecord(part.input) ? part.input : {};
    if (!id || !name) continue;
    toolUses.push({ id, name, input });
  }
  return toolUses;
}

export async function executeToolUse(params: {
  toolsByName: Map<string, AgentTool>;
  toolUse: ToolUse;
}): Promise<{ content: string; isError: boolean }> {
  const tool = params.toolsByName.get(params.toolUse.name);
  const execute = tool?.execute;
  if (!execute) {
    return {
      isError: true,
      content: `Tool not found: ${params.toolUse.name}`,
    };
  }

  try {
    const result = await execute(params.toolUse.id, params.toolUse.input);
    if (typeof result === "string") {
      return { content: result, isError: false };
    }
    if (isRecord(result) && typeof result.content === "string") {
      return { content: result.content, isError: false };
    }
    return {
      content: JSON.stringify(result ?? null),
      isError: false,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { content: message, isError: true };
  }
}

