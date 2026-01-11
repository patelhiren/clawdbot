import crypto from "node:crypto";
import os from "node:os";

import type { AgentTool } from "@mariozechner/pi-agent-core";
import type { SkillSnapshot } from "./skills.js";
import { resolveHeartbeatPrompt } from "../auto-reply/heartbeat.js";
import { isSilentReplyText, SILENT_REPLY_TOKEN } from "../auto-reply/tokens.js";
import type { ThinkLevel } from "../auto-reply/thinking.js";
import type { ClawdbotConfig } from "../config/config.js";
import { shouldLogVerbose } from "../globals.js";
import { createSubsystemLogger } from "../logging.js";
import { runCommandWithTimeout, spawnCommand } from "../process/exec.js";
import { resolveUserPath } from "../utils.js";
import { resolveSessionAgentIds } from "./agent-scope.js";
import { FailoverError, resolveFailoverStatus } from "./failover-error.js";
import {
  buildBootstrapContextFiles,
  classifyFailoverReason,
  type EmbeddedContextFile,
  isFailoverErrorMessage,
} from "./pi-embedded-helpers.js";
import type { EmbeddedPiRunResult } from "./pi-embedded-runner.js";
import { buildAgentSystemPrompt } from "./system-prompt.js";
import { parseClaudeCliStream } from "./claude-cli-stream.js";
import {
  buildStreamJsonToolResult,
  buildStreamJsonUserMessage,
  buildToolAllowlist,
  executeToolUse,
  extractToolUsesFromAssistantLine,
  parseStreamJsonLine,
} from "./claude-cli-tool-loop.js";
import { createClawdbotCodingTools } from "./pi-tools.js";
import { applySkillEnvOverrides, loadWorkspaceSkillEntries } from "./skills.js";

import {
  filterBootstrapFilesForSession,
  loadWorkspaceBootstrapFiles,
} from "./workspace.js";

const log = createSubsystemLogger("agent/claude-cli");
const CLAUDE_CLI_QUEUE_KEY = "global";
const CLAUDE_CLI_RUN_QUEUE = new Map<string, Promise<unknown>>();

function enqueueClaudeCliRun<T>(
  key: string,
  task: () => Promise<T>,
): Promise<T> {
  const prior = CLAUDE_CLI_RUN_QUEUE.get(key) ?? Promise.resolve();
  const chained = prior.catch(() => undefined).then(task);
  const tracked = chained.finally(() => {
    if (CLAUDE_CLI_RUN_QUEUE.get(key) === tracked) {
      CLAUDE_CLI_RUN_QUEUE.delete(key);
    }
  });
  CLAUDE_CLI_RUN_QUEUE.set(key, tracked);
  return chained;
}

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

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function normalizeClaudeSessionId(raw?: string): string {
  const trimmed = raw?.trim();
  if (trimmed && UUID_RE.test(trimmed)) return trimmed;
  return crypto.randomUUID();
}

function resolveUserTimezone(configured?: string): string {
  const trimmed = configured?.trim();
  if (trimmed) {
    try {
      new Intl.DateTimeFormat("en-US", { timeZone: trimmed }).format(
        new Date(),
      );
      return trimmed;
    } catch {
      // ignore invalid timezone
    }
  }
  const host = Intl.DateTimeFormat().resolvedOptions().timeZone;
  return host?.trim() || "UTC";
}

function formatUserTime(date: Date, timeZone: string): string | undefined {
  try {
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone,
      weekday: "long",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hourCycle: "h23",
    }).formatToParts(date);
    const map: Record<string, string> = {};
    for (const part of parts) {
      if (part.type !== "literal") map[part.type] = part.value;
    }
    if (
      !map.weekday ||
      !map.year ||
      !map.month ||
      !map.day ||
      !map.hour ||
      !map.minute
    ) {
      return undefined;
    }
    return `${map.weekday} ${map.year}-${map.month}-${map.day} ${map.hour}:${map.minute}`;
  } catch {
    return undefined;
  }
}

function buildModelAliasLines(cfg?: ClawdbotConfig) {
  const models = cfg?.agents?.defaults?.models ?? {};
  const entries: Array<{ alias: string; model: string }> = [];
  for (const [keyRaw, entryRaw] of Object.entries(models)) {
    const model = String(keyRaw ?? "").trim();
    if (!model) continue;
    const alias = String(
      (entryRaw as { alias?: string } | undefined)?.alias ?? "",
    ).trim();
    if (!alias) continue;
    entries.push({ alias, model });
  }
  return entries
    .sort((a, b) => a.alias.localeCompare(b.alias))
    .map((entry) => `- ${entry.alias}: ${entry.model}`);
}

function buildSystemPrompt(params: {
  workspaceDir: string;
  config?: ClawdbotConfig;
  defaultThinkLevel?: ThinkLevel;
  extraSystemPrompt?: string;
  ownerNumbers?: string[];
  heartbeatPrompt?: string;
  tools: AgentTool[];
  contextFiles?: EmbeddedContextFile[];
  modelDisplay: string;
}) {
  const userTimezone = resolveUserTimezone(
    params.config?.agents?.defaults?.userTimezone,
  );
  const userTime = formatUserTime(new Date(), userTimezone);
  return buildAgentSystemPrompt({
    workspaceDir: params.workspaceDir,
    defaultThinkLevel: params.defaultThinkLevel,
    extraSystemPrompt: params.extraSystemPrompt,
    ownerNumbers: params.ownerNumbers,
    reasoningTagHint: false,
    heartbeatPrompt: params.heartbeatPrompt,
    runtimeInfo: {
      host: "clawdbot",
      os: `${os.type()} ${os.release()}`,
      arch: os.arch(),
      node: process.version,
      model: params.modelDisplay,
    },
    toolNames: params.tools.map((tool) => tool.name),
    modelAliasLines: buildModelAliasLines(params.config),
    userTimezone,
    userTime,
    contextFiles: params.contextFiles,
  });
}

function normalizeClaudeCliModel(modelId: string): string {
  const trimmed = modelId.trim();
  if (!trimmed) return "opus";

  // Claude CLI expects shorthand model names (opus|sonnet|haiku). In Clawdbot we
  // often carry full ids like `anthropic/claude-opus-4-5` or aliases like `opus`.
  const lower = trimmed.toLowerCase();

  // Common full-id forms.
  if (lower.includes("claude-opus")) return "opus";
  if (lower.includes("claude-sonnet")) return "sonnet";
  if (lower.includes("claude-haiku")) return "haiku";

  // Alias / shorthand forms.
  if (lower.startsWith("opus")) return "opus";
  if (lower.startsWith("sonnet")) return "sonnet";
  if (lower.startsWith("haiku")) return "haiku";

  return trimmed;
}

function toUsage(raw: Record<string, unknown>): ClaudeCliUsage | undefined {
  const pick = (key: string) =>
    typeof raw[key] === "number" && raw[key] > 0
      ? (raw[key] as number)
      : undefined;
  const input = pick("input_tokens") ?? pick("inputTokens");
  const output = pick("output_tokens") ?? pick("outputTokens");
  const cacheRead = pick("cache_read_input_tokens") ?? pick("cacheRead");
  const cacheWrite = pick("cache_write_input_tokens") ?? pick("cacheWrite");
  const total = pick("total_tokens") ?? pick("total");
  if (!input && !output && !cacheRead && !cacheWrite && !total)
    return undefined;
  return { input, output, cacheRead, cacheWrite, total };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function collectText(value: unknown): string {
  if (!value) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value.map((entry) => collectText(entry)).join("");
  }
  if (!isRecord(value)) return "";
  if (typeof value.text === "string") return value.text;
  if (typeof value.content === "string") return value.content;
  if (Array.isArray(value.content)) {
    return value.content.map((entry) => collectText(entry)).join("");
  }
  if (isRecord(value.message)) return collectText(value.message);
  return "";
}

function parseClaudeCliJson(raw: string): ClaudeCliOutput | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  const sessionId =
    (typeof parsed.session_id === "string" && parsed.session_id) ||
    (typeof parsed.sessionId === "string" && parsed.sessionId) ||
    (typeof parsed.conversation_id === "string" && parsed.conversation_id) ||
    undefined;
  const usage = isRecord(parsed.usage) ? toUsage(parsed.usage) : undefined;
  const text =
    collectText(parsed.message) ||
    collectText(parsed.content) ||
    collectText(parsed.result) ||
    collectText(parsed);
  return { text: text.trim(), sessionId, usage };
}

async function runClaudeCliOnce(params: {
  prompt: string;
  workspaceDir: string;
  modelId: string;
  systemPrompt: string;
  timeoutMs: number;
  sessionId: string;
  toolsByName?: Map<string, AgentTool>;
  toolAllowlist?: string;
  onAgentEvent?: (evt: { stream: string; data: Record<string, unknown> }) => void;
  runId?: string;
}): Promise<ClaudeCliOutput> {
  const wantsStream = params.onAgentEvent !== undefined;

  // Only use the tool-loop mode when we actually have tools to run.
  // Otherwise, prefer the simple runner which reliably terminates.
  const hasTools = Boolean(params.toolsByName && (params.toolsByName.size ?? 0) > 0);
  if (
    wantsStream &&
    hasTools &&
    params.toolsByName &&
    params.toolAllowlist !== undefined
  ) {
    return await runClaudeCliWithToolLoop({
      prompt: params.prompt,
      workspaceDir: params.workspaceDir,
      modelId: params.modelId,
      systemPrompt: params.systemPrompt,
      timeoutMs: params.timeoutMs,
      sessionId: params.sessionId,
      toolsByName: params.toolsByName,
      toolAllowlist: params.toolAllowlist,
      onAgentEvent: params.onAgentEvent,
    });
  }

  const args = [
    "-p",
    "--output-format",
    wantsStream ? "stream-json" : "json",
    "--model",
    normalizeClaudeCliModel(params.modelId),
    "--append-system-prompt",
    params.systemPrompt,
    "--dangerously-skip-permissions",
    "--session-id",
    params.sessionId,
  ];
  if (wantsStream) {
    args.push("--verbose", "--include-partial-messages");
  }

  log.info(
    `claude-cli exec: model=${normalizeClaudeCliModel(params.modelId)} promptChars=${params.prompt.length} systemPromptChars=${params.systemPrompt.length} stream=${wantsStream}`,
  );
  if (process.env.CLAWDBOT_CLAUDE_CLI_LOG_OUTPUT === "1") {
    const logArgs: string[] = [];
    for (let i = 0; i < args.length; i += 1) {
      const arg = args[i];
      if (arg === "--append-system-prompt") {
        logArgs.push(arg, `<systemPrompt:${params.systemPrompt.length} chars>`);
        i += 1;
        continue;
      }
      if (arg === "--session-id") {
        logArgs.push(arg, args[i + 1] ?? "");
        i += 1;
        continue;
      }
      logArgs.push(arg);
    }
    log.info(`claude-cli argv: claude ${logArgs.join(" ")}`);
  }

  const result = await runCommandWithTimeout(["claude", ...args], {
    timeoutMs: params.timeoutMs,
    cwd: params.workspaceDir,
    input: params.prompt,
    env: (() => {
      const next = { ...process.env };
      delete next.ANTHROPIC_API_KEY;
      return next;
    })(),
  });
  if (process.env.CLAWDBOT_CLAUDE_CLI_LOG_OUTPUT === "1") {
    const stdoutDump = result.stdout.trim();
    const stderrDump = result.stderr.trim();
    if (stdoutDump) {
      log.info(`claude-cli stdout:\n${stdoutDump}`);
    }
    if (stderrDump) {
      log.info(`claude-cli stderr:\n${stderrDump}`);
    }
  }
  const stdout = result.stdout.trim();
  const logOutputText = process.env.CLAWDBOT_CLAUDE_CLI_LOG_OUTPUT === "1";
  if (shouldLogVerbose()) {
    if (stdout) {
      log.debug(`claude-cli stdout:\n${stdout}`);
    }
    if (result.stderr.trim()) {
      log.debug(`claude-cli stderr:\n${result.stderr.trim()}`);
    }
  }

  const maybeHandleNonzero = (out: string) => {
    if (result.code === 0) return;
    const err = result.stderr.trim() || out || "Claude CLI failed.";
    if (isFailoverErrorMessage(err)) {
      const reason = classifyFailoverReason(err) ?? "unknown";
      const status = resolveFailoverStatus(reason);
      throw new FailoverError(err, {
        reason,
        provider: "claude-cli",
        model: params.modelId,
        status,
      });
    }
    throw new Error(err);
  };

  if (!wantsStream) {
    maybeHandleNonzero(stdout);
    const parsed = parseClaudeCliJson(stdout);
    const output = parsed ?? { text: stdout };
    if (logOutputText) {
      const text = output.text?.trim();
      if (text) {
        log.info(`claude-cli output:\n${text}`);
      }
    }
    return output;
  }

  const { output, rawText } = parseClaudeCliStream({
    stdout,
    onAgentEvent: params.onAgentEvent,
  });
  maybeHandleNonzero(rawText);

  // Treat the silent-reply token as an intentional "no output" outcome,
  // not a runtime error.
  if (isSilentReplyText(output.text, SILENT_REPLY_TOKEN)) {
    return { ...output, text: "" };
  }

  if (logOutputText) {
    const text = output.text?.trim();
    if (text) {
      log.info(`claude-cli output:\n${text}`);
    }
  }
  return output;
}

async function runClaudeCliWithToolLoop(params: {
  prompt: string;
  workspaceDir: string;
  modelId: string;
  systemPrompt: string;
  timeoutMs: number;
  sessionId: string;
  toolsByName: Map<string, AgentTool>;
  toolAllowlist: string;
  onAgentEvent?: (evt: { stream: string; data: Record<string, unknown> }) => void;
}): Promise<ClaudeCliOutput> {
  const argv = [
    "claude",
    "-p",
    "--verbose",
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
    "--include-partial-messages",
    "--model",
    normalizeClaudeCliModel(params.modelId),
    "--append-system-prompt",
    params.systemPrompt,
    "--dangerously-skip-permissions",
    "--replay-user-messages",
    "--session-id",
    params.sessionId,
  ];
  if (params.toolAllowlist.trim()) {
    argv.push("--tools", params.toolAllowlist);
  } else {
    argv.push("--tools", "");
  }
  const env = (() => {
    const next = { ...process.env };
    delete next.ANTHROPIC_API_KEY;
    return next;
  })();

  const handle = spawnCommand(argv, { cwd: params.workspaceDir, env });
  const child = handle.child;
  const stdin = child.stdin;
  if (!stdin) throw new Error("Failed to open stdin for claude");

  const toWrite = buildStreamJsonUserMessage(params.prompt) + "\n";
  stdin.write(toWrite);

  let stdout = "";
  let stderr = "";
  const toolUsesSeen = new Set<string>();

  const maybeExecuteToolsFromLine = async (rawLine: string) => {
    const obj = parseStreamJsonLine(rawLine);
    if (!obj) return;
    for (const toolUse of extractToolUsesFromAssistantLine(obj)) {
      if (toolUsesSeen.has(toolUse.id)) continue;
      toolUsesSeen.add(toolUse.id);
      const result = await executeToolUse({
        toolsByName: params.toolsByName,
        toolUse,
      });
      stdin.write(
        buildStreamJsonToolResult({
          toolUseId: toolUse.id,
          content: result.content,
          isError: result.isError,
        }) + "\n",
      );
    }
  };

  let pending = "";
  let stdinClosed = false;
  child.stdout?.on("data", (d) => {
    const chunk = d.toString();
    stdout += chunk;
    pending += chunk;
    for (;;) {
      const idx = pending.indexOf("\n");
      if (idx < 0) break;
      const line = pending.slice(0, idx);
      pending = pending.slice(idx + 1);

      // Claude CLI will keep the stream open until stdin closes. Once we see the
      // final result envelope, close stdin so the process can exit.
      const parsedLine = parseStreamJsonLine(line);
      if (parsedLine?.type === "result" && stdin && !stdinClosed) {
        stdinClosed = true;
        stdin.end();
      }

      void maybeExecuteToolsFromLine(line);
    }
  });
  child.stderr?.on("data", (d) => {
    stderr += d.toString();
  });

  const timeout = setTimeout(() => {
    handle.kill("SIGKILL");
  }, params.timeoutMs);

  const result = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", (code, signal) => resolve({ code, signal }));
  });
  clearTimeout(timeout);

  // Emit parsed stream events (assistant/tool/lifecycle) using existing parser.
  const parsed = parseClaudeCliStream({ stdout, onAgentEvent: params.onAgentEvent });
  // Mirror runCommandWithTimeout signature for error handling/logging.
  if (result.code !== 0) {
    const err = stderr.trim() || parsed.rawText || "Claude CLI failed.";
    if (isFailoverErrorMessage(err)) {
      const reason = classifyFailoverReason(err) ?? "unknown";
      const status = resolveFailoverStatus(reason);
      throw new FailoverError(err, { reason, provider: "claude-cli", model: params.modelId, status });
    }
    throw new Error(err);
  }
  return parsed.output;
}

export async function runClaudeCliAgent(params: {
  sessionId: string;
  sessionKey?: string;
  sessionFile: string;
  workspaceDir: string;
  agentDir?: string;
  config?: ClawdbotConfig;
  skillsSnapshot?: SkillSnapshot;
  prompt: string;
  provider?: string;
  model?: string;
  thinkLevel?: ThinkLevel;
  timeoutMs: number;
  runId: string;
  extraSystemPrompt?: string;
  ownerNumbers?: string[];
  claudeSessionId?: string;
  onAgentEvent?: (evt: { stream: string; data: Record<string, unknown> }) => void;
}): Promise<EmbeddedPiRunResult> {
  const started = Date.now();
  const resolvedWorkspace = resolveUserPath(params.workspaceDir);
  const workspaceDir = resolvedWorkspace;

  params.onAgentEvent?.({
    stream: "lifecycle",
    data: {
      phase: "start",
      startedAt: started,
    },
  });

  const modelId = (params.model ?? "opus").trim() || "opus";
  const modelDisplay = `${params.provider ?? "claude-cli"}/${modelId}`;

  const extraSystemPrompt = [params.extraSystemPrompt?.trim()]
    .filter(Boolean)
    .join("\n");

  const bootstrapFiles = filterBootstrapFilesForSession(
    await loadWorkspaceBootstrapFiles(workspaceDir),
    params.sessionKey ?? params.sessionId,
  );
  const contextFiles = buildBootstrapContextFiles(bootstrapFiles);
  const { defaultAgentId, sessionAgentId } = resolveSessionAgentIds({
    sessionKey: params.sessionKey,
    config: params.config,
  });
  const heartbeatPrompt =
    sessionAgentId === defaultAgentId
      ? resolveHeartbeatPrompt(
          params.config?.agents?.defaults?.heartbeat?.prompt,
        )
      : undefined;
  const systemPrompt = buildSystemPrompt({
    workspaceDir,
    config: params.config,
    defaultThinkLevel: params.thinkLevel,
    extraSystemPrompt,
    ownerNumbers: params.ownerNumbers,
    heartbeatPrompt,
    tools: [],
    contextFiles,
    modelDisplay,
  });

  const resolvedAgentDir = params.agentDir ?? resolveUserPath("~/.clawdbot/agent");
  const skillEntries = !params.skillsSnapshot || !params.skillsSnapshot.resolvedSkills
    ? loadWorkspaceSkillEntries(workspaceDir)
    : [];
  const restoreSkillEnv = params.skillsSnapshot
    ? applySkillEnvOverrides({ skills: [], config: params.config })
    : applySkillEnvOverrides({ skills: skillEntries ?? [], config: params.config });
  try {
    const tools = createClawdbotCodingTools({
      bash: { ...params.config?.tools?.bash },
      workspaceDir,
      sessionKey: params.sessionKey ?? params.sessionId,
      agentDir: resolvedAgentDir,
      config: params.config,
      modelProvider: "anthropic",
    });
    const toolAllowlist = buildToolAllowlist(tools);
    const toolsByName = new Map<string, AgentTool>(
      tools.map((tool) => [tool.name, tool]),
    );

    const claudeSessionId = normalizeClaudeSessionId(params.claudeSessionId);
    const output = await enqueueClaudeCliRun(CLAUDE_CLI_QUEUE_KEY, () =>
      runClaudeCliOnce({
        prompt: params.prompt,
        workspaceDir,
        modelId,
        systemPrompt,
        timeoutMs: params.timeoutMs,
        sessionId: claudeSessionId,
        runId: params.runId,
        onAgentEvent: params.onAgentEvent,
        toolsByName,
        toolAllowlist,
      }),
    );

    const text = output.text?.trim();
    const payloads = text ? [{ text }] : undefined;

    const endedAt = Date.now();
    params.onAgentEvent?.({
      stream: "lifecycle",
      data: {
        phase: "end",
        startedAt: started,
        endedAt,
      },
    });

    return {
      payloads,
      meta: {
        durationMs: endedAt - started,
        agentMeta: {
          sessionId: output.sessionId ?? claudeSessionId,
          provider: params.provider ?? "claude-cli",
          model: modelId,
          usage: output.usage,
        },
      },
    };
  } catch (err) {
    const endedAt = Date.now();
    params.onAgentEvent?.({
      stream: "lifecycle",
      data: {
        phase: "error",
        startedAt: started,
        endedAt,
        error: String(err),
      },
    });
    throw err;
  } finally {
    restoreSkillEnv?.();
  }
}
