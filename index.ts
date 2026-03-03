/**
 * Vertex AI Anthropic Provider Extension for Pi
 *
 * Provides Claude models through Google Cloud Vertex AI.
 * Authentication via gcloud CLI (/login) or environment variables.
 */

import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Context,
	calculateCost,
	createAssistantMessageEventStream,
	type ImageContent,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type StopReason,
	type TextContent,
	type ThinkingContent,
	type Tool,
	type ToolCall,
	type ToolResultMessage,
} from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { execFileSync, execFile as execFileCb } from "node:child_process";
import { readFileSync } from "node:fs";
import { promisify } from "node:util";

const execFileAsync = promisify(execFileCb);

// =============================================================================
// Typed Anthropic API message structures
// =============================================================================

interface AnthropicTextBlock {
	type: "text";
	text: string;
	cache_control?: { type: "ephemeral" };
}

interface AnthropicThinkingBlock {
	type: "thinking";
	thinking: string;
	signature: string;
}

interface AnthropicToolUseBlock {
	type: "tool_use";
	id: string;
	name: string;
	input: Record<string, unknown>;
}

interface AnthropicToolResultBlock {
	type: "tool_result";
	tool_use_id: string;
	content: string | AnthropicContentBlock[];
	is_error?: boolean;
	cache_control?: { type: "ephemeral" };
}

interface AnthropicImageBlock {
	type: "image";
	source: { type: "base64"; media_type: string; data: string };
}

type AnthropicContentBlock =
	| AnthropicTextBlock
	| AnthropicThinkingBlock
	| AnthropicToolUseBlock
	| AnthropicToolResultBlock
	| AnthropicImageBlock;

interface AnthropicMessage {
	role: "user" | "assistant";
	content: string | AnthropicContentBlock[];
}

// =============================================================================
// Input Validation
// =============================================================================

/** Validate GCP project ID format. Accepts standard and org-prefixed formats. */
function validateProjectId(project: string): string {
	const sanitized = project.trim();
	// GCP project IDs: lowercase letters, digits, hyphens, 6-30 chars
	// Also allow org-prefixed: google.com:my-project
	if (!sanitized || !/^[a-z][a-z0-9.:_-]{3,48}[a-z0-9]$/i.test(sanitized)) {
		throw new Error(`Invalid GCP project ID format: "${sanitized}"`);
	}
	return sanitized;
}

/** Validate GCP region format. */
function validateRegion(region: string): string {
	const sanitized = region.trim();
	if (!sanitized || !/^(?:global|[a-z]+-[a-z]+\d+(-[a-z])?)$/.test(sanitized)) {
		throw new Error(`Invalid GCP region format: "${sanitized}"`);
	}
	return sanitized;
}

// =============================================================================
// Configuration (lazy evaluation)
// =============================================================================

/**
 * Configuration can be set via:
 * 1. Environment variables (highest priority)
 * 2. Persisted credentials from /login (stored in ~/.pi/agent/auth.json)
 * 3. Hardcoded defaults (fallback)
 */

function getPersistedCredentials(): { project?: string; region?: string } {
	try {
		const authPath = `${process.env.HOME}/.pi/agent/auth.json`;
		const data = JSON.parse(readFileSync(authPath, "utf-8"));
		const cred = data["vertex-anthropic"];
		if (cred?.type === "oauth") {
			return {
				project: cred.project,
				region: cred.region,
			};
		}
	} catch {}
	return {};
}

/** Cached gcloud path — discovered lazily on first use. */
let cachedGcloudPath: string | null = null;

function findGcloud(): string {
	if (cachedGcloudPath) return cachedGcloudPath;

	const paths = [
		"/usr/local/bin/gcloud",
		"/usr/bin/gcloud",
		`${process.env.HOME}/google-cloud-sdk/bin/gcloud`,
		"gcloud",
	];

	for (const p of paths) {
		try {
			execFileSync(p, ["version"], { stdio: "ignore", timeout: 2000 });
			cachedGcloudPath = p;
			return p;
		} catch {}
	}

	cachedGcloudPath = "gcloud";
	return "gcloud";
}

function getConfig() {
	const persisted = getPersistedCredentials();
	return {
		project: process.env.VERTEX_PROJECT_ID || persisted.project || "your-gcp-project-id",
		region: process.env.VERTEX_REGION || persisted.region || "us-east5",
		get gcloudPath() {
			return process.env.VERTEX_GCLOUD_PATH || findGcloud();
		},
	};
}

// =============================================================================
// Token Caching (async, non-blocking)
// =============================================================================

let tokenCache: { token: string; expiresAt: number } | null = null;

/** Get a gcloud access token, using cache when possible. Non-blocking. */
async function getAccessToken(gcloudPath: string): Promise<string> {
	if (tokenCache && Date.now() < tokenCache.expiresAt) {
		return tokenCache.token;
	}

	const { stdout } = await execFileAsync(gcloudPath, ["auth", "print-access-token"], {
		timeout: 10000,
	});
	const token = stdout.trim();

	if (!token || token.length < 20) {
		throw new Error("Invalid access token from gcloud. Run: gcloud auth login");
	}

	// gcloud tokens last ~60 min; cache for 50 min to avoid edge-case expiry
	tokenCache = { token, expiresAt: Date.now() + 50 * 60 * 1000 };
	return token;
}

/** Invalidate the cached token (e.g., on 401). */
function invalidateTokenCache(): void {
	tokenCache = null;
}

// =============================================================================
// Message Transformation (handles incomplete tool calls)
// =============================================================================

/**
 * Transform messages to handle incomplete tool calls and cross-provider compatibility.
 * This removes errored/aborted assistant messages and inserts synthetic tool results
 * for orphaned tool calls to prevent API errors.
 */
function transformMessages(
	messages: Message[],
	model: Model<Api>,
	normalizeToolCallIdFn?: (id: string) => string,
): Message[] {
	// Build a map of original tool call IDs to normalized IDs
	const toolCallIdMap = new Map<string, string>();

	// First pass: transform messages (thinking blocks, tool call ID normalization)
	const transformed = messages.map((msg) => {
		if (msg.role === "user") {
			return msg;
		}

		// Handle toolResult messages — normalize toolCallId if we have a mapping
		if (msg.role === "toolResult") {
			const normalizedId = toolCallIdMap.get(msg.toolCallId);
			if (normalizedId && normalizedId !== msg.toolCallId) {
				return { ...msg, toolCallId: normalizedId };
			}
			return msg;
		}

		// Assistant messages need transformation check
		if (msg.role === "assistant") {
			const assistantMsg = msg as AssistantMessage;
			const isSameModel =
				assistantMsg.provider === model.provider &&
				assistantMsg.api === model.api &&
				assistantMsg.model === model.id;

			const transformedContent = assistantMsg.content.flatMap((block): (TextContent | ThinkingContent | ToolCall)[] => {
				if (block.type === "thinking") {
					const thinkingBlock = block as ThinkingContent;
					if (isSameModel && thinkingBlock.thinkingSignature) return [thinkingBlock];
					if (!thinkingBlock.thinking || thinkingBlock.thinking.trim() === "") return [];
					if (isSameModel) return [thinkingBlock];
					return [{ type: "text" as const, text: thinkingBlock.thinking } as TextContent];
				}

				if (block.type === "text") {
					if (isSameModel) return [block as TextContent];
					return [{ type: "text" as const, text: block.text } as TextContent];
				}

				if (block.type === "toolCall") {
					const toolCall = block as ToolCall;
					if (!isSameModel && normalizeToolCallIdFn) {
						const normalizedId = normalizeToolCallIdFn(toolCall.id);
						if (normalizedId !== toolCall.id) {
							toolCallIdMap.set(toolCall.id, normalizedId);
							return [{ ...toolCall, id: normalizedId }];
						}
					}
					return [toolCall];
				}

				return [block];
			});

			return { ...assistantMsg, content: transformedContent } as AssistantMessage;
		}

		return msg;
	});

	// Second pass: insert synthetic empty tool results for orphaned tool calls
	const result: Message[] = [];
	let pendingToolCalls: ToolCall[] = [];
	let existingToolResultIds = new Set<string>();
	const skippedToolCallIds = new Set<string>();

	for (let i = 0; i < transformed.length; i++) {
		const msg = transformed[i];

		if (msg.role === "assistant") {
			// Insert synthetic results for any orphaned tool calls from previous assistant
			if (pendingToolCalls.length > 0) {
				for (const tc of pendingToolCalls) {
					if (!existingToolResultIds.has(tc.id)) {
						result.push({
							role: "toolResult",
							toolCallId: tc.id,
							toolName: tc.name,
							content: [{ type: "text", text: "No result provided" }],
							isError: true,
							timestamp: Date.now(),
						} as ToolResultMessage);
					}
				}
			}

			const assistantMsg = msg as AssistantMessage;

			// Skip errored/aborted assistant messages entirely
			if (assistantMsg.stopReason === "error" || assistantMsg.stopReason === "aborted") {
				const toolCalls = assistantMsg.content.filter((b) => b.type === "toolCall") as ToolCall[];
				for (const tc of toolCalls) {
					skippedToolCallIds.add(tc.id);
				}
				// Reset pending state since we're skipping this message
				pendingToolCalls = [];
				existingToolResultIds = new Set();
				continue;
			}

			// Always reset pending state for each new (non-errored) assistant message
			const toolCalls = assistantMsg.content.filter((b) => b.type === "toolCall") as ToolCall[];
			pendingToolCalls = toolCalls;
			existingToolResultIds = new Set();

			result.push(msg);
		} else if (msg.role === "toolResult") {
			const toolResultMsg = msg as ToolResultMessage;
			// Drop tool results whose assistant tool_use was skipped (errored/aborted)
			if (skippedToolCallIds.has(toolResultMsg.toolCallId)) {
				continue;
			}
			existingToolResultIds.add(toolResultMsg.toolCallId);
			result.push(msg);
		} else if (msg.role === "user") {
			// User message interrupts tool flow — insert synthetic results for orphaned calls
			if (pendingToolCalls.length > 0) {
				for (const tc of pendingToolCalls) {
					if (!existingToolResultIds.has(tc.id)) {
						result.push({
							role: "toolResult",
							toolCallId: tc.id,
							toolName: tc.name,
							content: [{ type: "text", text: "No result provided" }],
							isError: true,
							timestamp: Date.now(),
						} as ToolResultMessage);
					}
				}
				pendingToolCalls = [];
				existingToolResultIds = new Set();
			}
			result.push(msg);
		} else {
			result.push(msg);
		}
	}

	return result;
}

// =============================================================================
// Message conversion
// =============================================================================

function sanitizeSurrogates(text: string): string {
	return text.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
}

function convertContentBlocks(
	content: (TextContent | ImageContent)[],
): string | Array<AnthropicTextBlock | AnthropicImageBlock> {
	const hasImages = content.some((c) => c.type === "image");
	if (!hasImages) {
		return sanitizeSurrogates(content.map((c) => (c as TextContent).text).join("\n"));
	}

	const blocks: Array<AnthropicTextBlock | AnthropicImageBlock> = content.map((block) => {
		if (block.type === "text") {
			return { type: "text" as const, text: sanitizeSurrogates(block.text) };
		}
		return {
			type: "image" as const,
			source: {
				type: "base64" as const,
				media_type: block.mimeType,
				data: block.data,
			},
		};
	});

	if (!blocks.some((b) => b.type === "text")) {
		blocks.unshift({ type: "text" as const, text: "(see attached image)" });
	}

	return blocks;
}

/** Normalize tool call IDs to match Anthropic's required pattern and length. */
function normalizeToolCallId(id: string): string {
	return id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
}

function convertMessages(messages: Message[], model: Model<Api>, _tools?: Tool[]): AnthropicMessage[] {
	const params: AnthropicMessage[] = [];

	const transformedMessages = transformMessages(messages, model, normalizeToolCallId);

	for (let i = 0; i < transformedMessages.length; i++) {
		const msg = transformedMessages[i];

		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				if (msg.content.trim()) {
					params.push({ role: "user", content: sanitizeSurrogates(msg.content) });
				}
			} else {
				const blocks: AnthropicContentBlock[] = msg.content.map((item) =>
					item.type === "text"
						? { type: "text" as const, text: sanitizeSurrogates(item.text) }
						: {
								type: "image" as const,
								source: { type: "base64" as const, media_type: item.mimeType, data: item.data },
							},
				);
				if (blocks.length > 0) {
					params.push({ role: "user", content: blocks });
				}
			}
		} else if (msg.role === "assistant") {
			const blocks: AnthropicContentBlock[] = [];
			for (const block of msg.content) {
				if (block.type === "text" && block.text.trim()) {
					blocks.push({ type: "text", text: sanitizeSurrogates(block.text) });
				} else if (block.type === "thinking" && block.thinking.trim()) {
					if ((block as ThinkingContent).thinkingSignature) {
						blocks.push({
							type: "thinking",
							thinking: sanitizeSurrogates(block.thinking),
							signature: (block as ThinkingContent).thinkingSignature!,
						});
					} else {
						blocks.push({ type: "text", text: sanitizeSurrogates(block.thinking) });
					}
				} else if (block.type === "toolCall") {
					blocks.push({
						type: "tool_use",
						id: block.id,
						name: block.name,
						input: block.arguments as Record<string, unknown>,
					});
				}
			}
			if (blocks.length > 0) {
				params.push({ role: "assistant", content: blocks });
			}
		} else if (msg.role === "toolResult") {
			const toolResults: AnthropicToolResultBlock[] = [];
			toolResults.push({
				type: "tool_result",
				tool_use_id: msg.toolCallId,
				content: convertContentBlocks(msg.content),
				is_error: msg.isError,
			});

			let j = i + 1;
			while (j < transformedMessages.length && transformedMessages[j].role === "toolResult") {
				const nextMsg = transformedMessages[j] as ToolResultMessage;
				toolResults.push({
					type: "tool_result",
					tool_use_id: nextMsg.toolCallId,
					content: convertContentBlocks(nextMsg.content),
					is_error: nextMsg.isError,
				});
				j++;
			}
			i = j - 1;
			params.push({ role: "user", content: toolResults });
		}
	}

	// -------------------------------------------------------------------------
	// Validation pass 1: ensure every tool_result references a tool_use in the
	// immediately preceding assistant message.
	// -------------------------------------------------------------------------
	const validated: AnthropicMessage[] = [];
	for (const msg of params) {
		if (
			msg.role === "user" &&
			Array.isArray(msg.content) &&
			(msg.content as AnthropicContentBlock[]).some((b) => b.type === "tool_result")
		) {
			const prevAssistant =
				validated.length > 0 && validated[validated.length - 1].role === "assistant"
					? validated[validated.length - 1]
					: null;

			const validToolUseIds = new Set<string>();
			if (prevAssistant && Array.isArray(prevAssistant.content)) {
				for (const block of prevAssistant.content as AnthropicContentBlock[]) {
					if (block.type === "tool_use") {
						validToolUseIds.add(block.id);
					}
				}
			}

			const filteredContent = (msg.content as AnthropicContentBlock[]).filter((block) => {
				if (block.type === "tool_result") {
					return validToolUseIds.has(block.tool_use_id);
				}
				return true;
			});

			if (filteredContent.length > 0) {
				validated.push({ ...msg, content: filteredContent });
			}
		} else {
			validated.push(msg);
		}
	}

	// -------------------------------------------------------------------------
	// Validation pass 2: ensure every tool_use has a corresponding tool_result
	// in the immediately following user message.
	// -------------------------------------------------------------------------
	for (let i = 0; i < validated.length; i++) {
		const msg = validated[i];
		if (msg.role !== "assistant" || !Array.isArray(msg.content)) continue;

		const toolUseIds = (msg.content as AnthropicContentBlock[])
			.filter((b): b is AnthropicToolUseBlock => b.type === "tool_use")
			.map((b) => b.id);
		if (toolUseIds.length === 0) continue;

		const nextMsg = i + 1 < validated.length ? validated[i + 1] : null;
		const existingResultIds = new Set<string>();
		if (nextMsg && nextMsg.role === "user" && Array.isArray(nextMsg.content)) {
			for (const block of nextMsg.content as AnthropicContentBlock[]) {
				if (block.type === "tool_result") {
					existingResultIds.add(block.tool_use_id);
				}
			}
		}

		const missingIds = toolUseIds.filter((id) => !existingResultIds.has(id));
		if (missingIds.length > 0) {
			const syntheticResults: AnthropicToolResultBlock[] = missingIds.map((id) => ({
				type: "tool_result" as const,
				tool_use_id: id,
				content: "No result provided",
				is_error: true,
			}));

			if (nextMsg && nextMsg.role === "user" && Array.isArray(nextMsg.content)) {
				nextMsg.content = [...(nextMsg.content as AnthropicContentBlock[]), ...syntheticResults];
			} else {
				validated.splice(i + 1, 0, { role: "user", content: syntheticResults });
			}
		}
	}

	// -------------------------------------------------------------------------
	// Merge consecutive same-role messages (can happen after dropping messages).
	// Guard: never merge assistant messages containing thinking blocks, as
	// thinking signatures are tied to specific message contexts.
	// -------------------------------------------------------------------------
	const merged: AnthropicMessage[] = [];
	for (const msg of validated) {
		const prev = merged.length > 0 ? merged[merged.length - 1] : null;
		if (prev && prev.role === msg.role && prev.role === "user") {
			if (typeof prev.content === "string" && typeof msg.content === "string") {
				prev.content = prev.content + "\n" + msg.content;
			} else {
				const prevBlocks: AnthropicContentBlock[] =
					typeof prev.content === "string"
						? [{ type: "text", text: prev.content }]
						: (prev.content as AnthropicContentBlock[]);
				const msgBlocks: AnthropicContentBlock[] =
					typeof msg.content === "string"
						? [{ type: "text", text: msg.content }]
						: (msg.content as AnthropicContentBlock[]);
				prev.content = [...prevBlocks, ...msgBlocks];
			}
		} else if (prev && prev.role === msg.role && prev.role === "assistant") {
			// Don't merge assistant messages that contain thinking blocks
			const prevHasThinking = Array.isArray(prev.content) &&
				(prev.content as AnthropicContentBlock[]).some((b) => b.type === "thinking");
			const msgHasThinking = Array.isArray(msg.content) &&
				(msg.content as AnthropicContentBlock[]).some((b) => b.type === "thinking");

			if (prevHasThinking || msgHasThinking) {
				// Insert a separator user message to maintain alternation
				merged.push({ role: "user", content: "(continued)" });
				merged.push(msg);
			} else {
				prev.content = [
					...(prev.content as AnthropicContentBlock[]),
					...(msg.content as AnthropicContentBlock[]),
				];
			}
		} else {
			merged.push(msg);
		}
	}

	// Ensure conversation starts with user message (Anthropic requirement)
	while (merged.length > 0 && merged[0].role !== "user") {
		merged.shift();
	}

	// Ensure strict user/assistant alternation by inserting placeholder messages
	const alternated: AnthropicMessage[] = [];
	for (const msg of merged) {
		const prev = alternated.length > 0 ? alternated[alternated.length - 1] : null;
		if (prev && prev.role === msg.role) {
			if (msg.role === "user") {
				alternated.push({ role: "assistant", content: [{ type: "text", text: "(continued)" }] });
			} else {
				alternated.push({ role: "user", content: "(continued)" });
			}
		}
		alternated.push(msg);
	}

	// Add cache control to last user message
	if (alternated.length > 0) {
		const last = alternated[alternated.length - 1];
		if (last.role === "user" && Array.isArray(last.content)) {
			const blocks = last.content as AnthropicContentBlock[];
			const lastBlock = blocks[blocks.length - 1];
			if (lastBlock && "cache_control" in lastBlock || lastBlock?.type === "text" || lastBlock?.type === "tool_result") {
				(lastBlock as AnthropicTextBlock | AnthropicToolResultBlock).cache_control = { type: "ephemeral" };
			}
		}
	}

	return alternated;
}

function convertTools(tools: Tool[]): Array<{
	name: string;
	description: string;
	input_schema: { type: "object"; properties: Record<string, unknown>; required: string[] };
}> {
	return tools.map((tool) => ({
		name: tool.name,
		description: tool.description,
		input_schema: {
			type: "object" as const,
			properties: (tool.parameters as Record<string, unknown>)?.properties as Record<string, unknown> || {},
			required: (tool.parameters as Record<string, unknown>)?.required as string[] || [],
		},
	}));
}

function mapStopReason(reason: string): StopReason {
	switch (reason) {
		case "end_turn":
		case "pause_turn":
		case "stop_sequence":
			return "stop";
		case "max_tokens":
			return "length";
		case "tool_use":
			return "toolUse";
		default:
			return "error";
	}
}

// =============================================================================
// SSE Parser for Vertex AI streamRawPredict
// =============================================================================

async function* parseSSE(response: Response): AsyncGenerator<Record<string, unknown>> {
	const body = response.body;
	if (!body) {
		throw new Error("Response body is null — cannot stream SSE");
	}

	const reader = body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";

	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split("\n");
			buffer = lines.pop()!; // Keep incomplete line in buffer

			let eventType = "";
			let dataLines: string[] = [];

			for (const line of lines) {
				if (line.startsWith("event: ")) {
					eventType = line.slice(7).trim();
				} else if (line.startsWith("data: ")) {
					// SSE spec: multiple data lines are concatenated with newlines
					dataLines.push(line.slice(6));
				} else if (line === "" && dataLines.length > 0) {
					const data = dataLines.join("\n").trim();
					dataLines = [];
					try {
						const parsed = JSON.parse(data);
						parsed._eventType = eventType;
						yield parsed;
					} catch (parseError) {
						// Check if this is an error event from the server
						if (eventType === "error") {
							throw new Error(`Vertex AI SSE error: ${data}`);
						}
						// Log non-error parse failures for debugging (non-fatal)
						console.error(`[vertex-anthropic] Failed to parse SSE data: ${data.slice(0, 200)}`);
					}
					eventType = "";
				}
			}
		}

		// Handle any remaining data in buffer after stream ends
		if (buffer.trim()) {
			try {
				const parsed = JSON.parse(buffer.trim());
				parsed._eventType = "";
				yield parsed;
			} catch {
				// Incomplete data at end of stream — non-fatal
			}
		}
	} finally {
		reader.releaseLock();
	}
}

// =============================================================================
// Retry Logic
// =============================================================================

interface FetchWithRetryOptions {
	maxRetries?: number;
	signal?: AbortSignal;
	onRetry?: (attempt: number, status: number, delay: number) => void;
}

async function fetchWithRetry(
	url: string,
	init: RequestInit,
	options: FetchWithRetryOptions = {},
): Promise<Response> {
	const { maxRetries = 3, signal, onRetry } = options;
	let lastError: Error | null = null;

	for (let attempt = 0; attempt <= maxRetries; attempt++) {
		if (signal?.aborted) {
			throw new Error("Request was aborted");
		}

		try {
			const response = await fetch(url, { ...init, signal });

			// Don't retry on success or client errors (except 429)
			if (response.ok || (response.status < 500 && response.status !== 429)) {
				return response;
			}

			// Retry on 429 (rate limit) and 5xx (server errors)
			if (attempt < maxRetries) {
				const retryAfterHeader = response.headers.get("retry-after");
				const delay = retryAfterHeader
					? Math.min(parseInt(retryAfterHeader, 10) * 1000, 30000)
					: Math.min(1000 * Math.pow(2, attempt) + Math.random() * 500, 30000);

				onRetry?.(attempt + 1, response.status, delay);
				await new Promise((resolve) => setTimeout(resolve, delay));
				continue;
			}

			// Exhausted retries — return the error response for the caller to handle
			return response;
		} catch (error) {
			lastError = error instanceof Error ? error : new Error(String(error));

			// Don't retry on abort
			if (signal?.aborted || lastError.name === "AbortError") {
				throw lastError;
			}

			// Retry network errors
			if (attempt < maxRetries) {
				const delay = Math.min(1000 * Math.pow(2, attempt) + Math.random() * 500, 30000);
				onRetry?.(attempt + 1, 0, delay);
				await new Promise((resolve) => setTimeout(resolve, delay));
				continue;
			}
		}
	}

	throw lastError || new Error("Request failed after retries");
}

// =============================================================================
// Streaming Implementation
// =============================================================================

function streamVertexAnthropic(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	let streamEnded = false;

	/** Safely push an end/error event and close the stream. */
	function safeEnd(event: Parameters<typeof stream.push>[0]): void {
		if (streamEnded) return;
		streamEnded = true;
		stream.push(event);
		stream.end();
	}

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const config = getConfig();
			const project = (model as { project?: string }).project || config.project;
			const region = (model as { region?: string }).region || config.region;
			const gcloudPath = (model as { gcloudPath?: string }).gcloudPath || config.gcloudPath;

			// Get access token (async, cached)
			let token = await getAccessToken(gcloudPath);

			// Build Anthropic Messages API request body
			const body: Record<string, unknown> = {
				anthropic_version: "vertex-2023-10-16",
				messages: convertMessages(context.messages, model, context.tools),
				max_tokens: options?.maxTokens || Math.floor(model.maxTokens / 3),
				stream: true,
			};

			if (context.systemPrompt) {
				body.system = [
					{
						type: "text",
						text: sanitizeSurrogates(context.systemPrompt),
						cache_control: { type: "ephemeral" },
					},
				];
			}

			if (context.tools) {
				body.tools = convertTools(context.tools);
			}

			// Handle thinking/reasoning
			if (options?.reasoning && model.reasoning) {
				// Adaptive thinking is supported on Claude 4.6+ models
				const isAdaptiveSupported = model.id.includes("-4-6");

				const customBudget = options.thinkingBudgets?.[options.reasoning as keyof typeof options.thinkingBudgets];

				// Provide plenty of max_tokens for xhigh
				if (options.reasoning === "xhigh") {
					body.max_tokens = Math.max(body.max_tokens as number, 64000);
				}

				if (customBudget) {
					body.thinking = { type: "enabled", budget_tokens: customBudget };
					body.max_tokens = Math.max(body.max_tokens as number, customBudget + 1024);
				} else if (isAdaptiveSupported) {
					let effort = "high";
					if (options.reasoning === "xhigh") effort = "max";
					else if (options.reasoning === "high") effort = "high";
					else if (options.reasoning === "medium") effort = "medium";
					else if (options.reasoning === "low" || options.reasoning === "minimal") effort = "low";

					body.thinking = { type: "adaptive" };
					body.output_config = { effort };
				} else {
					const defaultBudgets: Record<string, number> = {
						minimal: 1024,
						low: 4096,
						medium: 10240,
						high: 20480,
						xhigh: 40000,
					};
					const budget = defaultBudgets[options.reasoning] ?? 10240;
					body.thinking = { type: "enabled", budget_tokens: budget };
					body.max_tokens = Math.max(body.max_tokens as number, budget + 1024);
					if ((body.max_tokens as number) > model.maxTokens) {
						body.max_tokens = model.maxTokens;
						(body.thinking as Record<string, unknown>).budget_tokens = Math.max(1024, model.maxTokens - 1024);
					}
				}
			}

			// Build Vertex AI endpoint URL
			const vertexModelId = (model as { vertexModelId?: string }).vertexModelId || model.id;

			const endpoint =
				region === "global"
					? "aiplatform.googleapis.com"
					: `${region}-aiplatform.googleapis.com`;

			const url = `https://${endpoint}/v1/projects/${project}/locations/${region}/publishers/anthropic/models/${vertexModelId}:streamRawPredict`;

			// Make request with retry logic
			const response = await fetchWithRetry(
				url,
				{
					method: "POST",
					headers: {
						Authorization: `Bearer ${token}`,
						"Content-Type": "application/json",
					},
					body: JSON.stringify(body),
				},
				{
					maxRetries: 3,
					signal: options?.signal,
					onRetry: (attempt, status, delay) => {
						// On 401, invalidate token cache so next retry gets a fresh token
						if (status === 401) {
							invalidateTokenCache();
							// Re-fetch token synchronously for the retry
							// (the retry loop will use the new token via the Authorization header)
						}
						console.error(
							`[vertex-anthropic] Retry ${attempt}: status=${status}, delay=${Math.round(delay)}ms`,
						);
					},
				},
			);

			if (!response.ok) {
				const errorText = await response.text();
				throw new Error(`Vertex AI error (${response.status}): ${errorText}`);
			}

			stream.push({ type: "start", partial: output });

			// Use a Map for O(1) lookup instead of findIndex
			type StreamBlock = (ThinkingContent | TextContent | (ToolCall & { partialJson: string })) & {
				index: number;
			};
			const blocksByStreamIndex = new Map<number, { arrayIndex: number; block: StreamBlock }>();

			for await (const event of parseSSE(response)) {
				const eventType = event.type as string;

				if (eventType === "message_start") {
					const usage = (event.message as Record<string, unknown>)?.usage as Record<string, number> | undefined;
					if (usage) {
						output.usage.input = usage.input_tokens || 0;
						output.usage.output = usage.output_tokens || 0;
						output.usage.cacheRead = usage.cache_read_input_tokens || 0;
						output.usage.cacheWrite = usage.cache_creation_input_tokens || 0;
						output.usage.totalTokens =
							output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
						calculateCost(model, output.usage);
					}
				} else if (eventType === "content_block_start") {
					const contentBlock = event.content_block as Record<string, string>;
					const streamIndex = event.index as number;

					if (contentBlock.type === "text") {
						const block = { type: "text" as const, text: "", index: streamIndex } as StreamBlock;
						output.content.push(block as TextContent);
						const arrayIndex = output.content.length - 1;
						blocksByStreamIndex.set(streamIndex, { arrayIndex, block });
						stream.push({ type: "text_start", contentIndex: arrayIndex, partial: output });
					} else if (contentBlock.type === "thinking") {
						const block = {
							type: "thinking" as const,
							thinking: "",
							thinkingSignature: "",
							index: streamIndex,
						} as StreamBlock;
						output.content.push(block as ThinkingContent);
						const arrayIndex = output.content.length - 1;
						blocksByStreamIndex.set(streamIndex, { arrayIndex, block });
						stream.push({ type: "thinking_start", contentIndex: arrayIndex, partial: output });
					} else if (contentBlock.type === "tool_use") {
						const block = {
							type: "toolCall" as const,
							id: contentBlock.id,
							name: contentBlock.name,
							arguments: {},
							partialJson: "",
							index: streamIndex,
						} as StreamBlock;
						output.content.push(block as unknown as ToolCall);
						const arrayIndex = output.content.length - 1;
						blocksByStreamIndex.set(streamIndex, { arrayIndex, block });
						stream.push({ type: "toolcall_start", contentIndex: arrayIndex, partial: output });
					}
				} else if (eventType === "content_block_delta") {
					const entry = blocksByStreamIndex.get(event.index as number);
					if (!entry) continue;
					const { arrayIndex, block } = entry;
					const delta = event.delta as Record<string, string>;

					if (delta.type === "text_delta" && block.type === "text") {
						block.text += delta.text;
						stream.push({ type: "text_delta", contentIndex: arrayIndex, delta: delta.text, partial: output });
					} else if (delta.type === "thinking_delta" && block.type === "thinking") {
						block.thinking += delta.thinking;
						stream.push({
							type: "thinking_delta",
							contentIndex: arrayIndex,
							delta: delta.thinking,
							partial: output,
						});
					} else if (delta.type === "input_json_delta" && block.type === "toolCall") {
						(block as StreamBlock & { partialJson: string }).partialJson += delta.partial_json;
						try {
							block.arguments = JSON.parse(
								(block as StreamBlock & { partialJson: string }).partialJson,
							);
						} catch {}
						stream.push({
							type: "toolcall_delta",
							contentIndex: arrayIndex,
							delta: delta.partial_json,
							partial: output,
						});
					} else if (delta.type === "signature_delta" && block.type === "thinking") {
						block.thinkingSignature = (block.thinkingSignature || "") + delta.signature;
					}
				} else if (eventType === "content_block_stop") {
					const entry = blocksByStreamIndex.get(event.index as number);
					if (!entry) continue;
					const { arrayIndex, block } = entry;

					delete (block as any).index;
					if (block.type === "text") {
						stream.push({ type: "text_end", contentIndex: arrayIndex, content: block.text, partial: output });
					} else if (block.type === "thinking") {
						stream.push({
							type: "thinking_end",
							contentIndex: arrayIndex,
							content: block.thinking,
							partial: output,
						});
					} else if (block.type === "toolCall") {
						try {
							block.arguments = JSON.parse(
								(block as any).partialJson,
							);
						} catch {}
						delete (block as any).partialJson;
						stream.push({ type: "toolcall_end", contentIndex: arrayIndex, toolCall: block, partial: output });
					}
				} else if (eventType === "message_delta") {
					const delta = event.delta as Record<string, string> | undefined;
					if (delta?.stop_reason) {
						output.stopReason = mapStopReason(delta.stop_reason);
					}
					const usage = event.usage as Record<string, number> | undefined;
					if (usage) {
						output.usage.output = usage.output_tokens || output.usage.output;
						output.usage.totalTokens =
							output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
						calculateCost(model, output.usage);
					}
				} else if (eventType === "error") {
					const error = event.error as Record<string, string> | undefined;
					throw new Error(
						`Vertex AI stream error: ${error?.message || JSON.stringify(event)}`,
					);
				}
			}

			if (options?.signal?.aborted) {
				throw new Error("Request was aborted");
			}

			// Clean up internal properties from content blocks
			for (const block of output.content) {
				delete (block as any).index;
			}

			safeEnd({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse", message: output });
		} catch (error) {
			for (const block of output.content) {
				delete (block as any).index;
				delete (block as any).partialJson;
			}
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
			safeEnd({ type: "error", reason: output.stopReason, error: output });
		}
	})();

	return stream;
}

// =============================================================================
// Login Flow Helpers
// =============================================================================

function checkGcloudInstalled(gcloudPath: string): boolean {
	try {
		execFileSync(gcloudPath, ["version"], { stdio: "ignore", timeout: 2000 });
		return true;
	} catch {
		return false;
	}
}

function checkGcloudAuthenticated(gcloudPath: string): boolean {
	try {
		const token = execFileSync(gcloudPath, ["auth", "print-access-token"], {
			encoding: "utf-8",
			timeout: 5000,
			stdio: ["ignore", "pipe", "ignore"],
		}).trim();
		return !!token && !token.includes("ERROR") && token.length >= 20;
	} catch {
		return false;
	}
}

function getCurrentProject(gcloudPath: string): string | null {
	try {
		const project = execFileSync(gcloudPath, ["config", "get-value", "project"], {
			encoding: "utf-8",
			stdio: ["ignore", "pipe", "ignore"],
		}).trim();
		return project && project !== "(unset)" ? project : null;
	} catch {
		return null;
	}
}

function listProjects(gcloudPath: string): string[] {
	try {
		return execFileSync(gcloudPath, ["projects", "list", "--format=value(projectId)"], {
			encoding: "utf-8",
			stdio: ["ignore", "pipe", "ignore"],
			timeout: 10000,
		})
			.trim()
			.split("\n")
			.filter((p) => p && p !== "(unset)");
	} catch {
		return [];
	}
}

function isVertexApiEnabled(gcloudPath: string): boolean {
	try {
		const enabled = execFileSync(
			gcloudPath,
			["services", "list", "--enabled", "--filter=name:aiplatform.googleapis.com", "--format=value(name)"],
			{ encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"] },
		).trim();
		return !!enabled;
	} catch {
		return false;
	}
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default function (pi: ExtensionAPI) {
	const config = getConfig();

	const endpoint =
		config.region === "global" ? "aiplatform.googleapis.com" : `${config.region}-aiplatform.googleapis.com`;

	pi.registerProvider("vertex-anthropic", {
		baseUrl: `https://${endpoint}`,
		api: "anthropic-messages",

		oauth: {
			name: "Google Cloud Vertex AI (gcloud)",
			async login(callbacks) {
				callbacks.onAuth({ type: "progress", message: "Setting up Google Cloud Vertex AI..." });

				// Step 1: Check gcloud CLI
				let gcloudPath = config.gcloudPath;
				if (!checkGcloudInstalled(gcloudPath)) {
					const install = await callbacks.onPrompt({
						message:
							"gcloud CLI not found. Install Google Cloud SDK?\n\n" +
							"This will download and install gcloud from:\n" +
							"https://cloud.google.com/sdk/docs/install\n\n" +
							"(y/n)",
					});

					if (install?.toLowerCase() === "y") {
						callbacks.onAuth({ type: "info", message: "Opening installation guide in browser..." });
						callbacks.onAuth({ url: "https://cloud.google.com/sdk/docs/install" });
						throw new Error("Please install gcloud CLI and run /login again");
					} else {
						throw new Error("gcloud CLI required. Install from: https://cloud.google.com/sdk/docs/install");
					}
				}

				// Step 2: Check authentication
				callbacks.onAuth({ type: "progress", message: "Checking gcloud authentication..." });
				if (!checkGcloudAuthenticated(gcloudPath)) {
					const doAuth = await callbacks.onPrompt({
						message: "Not authenticated with gcloud. Run 'gcloud auth login' now? (y/n)",
					});

					if (doAuth?.toLowerCase() === "y") {
						callbacks.onAuth({ type: "progress", message: "Running gcloud auth login..." });
						callbacks.onAuth({ type: "info", message: "A browser window will open for authentication" });

						try {
							execFileSync(gcloudPath, ["auth", "login"], { stdio: "inherit" });
						} catch {
							throw new Error("Authentication failed. Please try: gcloud auth login");
						}
					} else {
						throw new Error("Authentication required. Run: gcloud auth login");
					}
				}

				// Step 3: Get/Set project
				callbacks.onAuth({ type: "progress", message: "Configuring project..." });
				let project = process.env.VERTEX_PROJECT_ID;

				if (!project || project === "your-gcp-project-id") {
					const currentProject = getCurrentProject(gcloudPath);
					if (currentProject) {
						const use = await callbacks.onPrompt({
							message: `Use current project '${currentProject}'? (y/n)`,
						});
						if (use?.toLowerCase() === "y") {
							project = currentProject;
						}
					}

					if (!project) {
						let projectPrompt = "Enter GCP project ID:";
						const projects = listProjects(gcloudPath);
						if (projects.length > 0 && projects.length < 20) {
							projectPrompt = `Available projects:\n${projects.map((p) => `  - ${p}`).join("\n")}\n\nEnter project ID:`;
						}

						const projectInput = await callbacks.onPrompt({ message: projectPrompt });
						if (!projectInput || projectInput.trim() === "") {
							throw new Error("Project ID required");
						}
						project = validateProjectId(projectInput);

						try {
							execFileSync(gcloudPath, ["config", "set", "project", project], { stdio: "ignore" });
						} catch {}
					}
				}

				process.env.VERTEX_PROJECT_ID = project;

				// Step 4: Select region
				callbacks.onAuth({ type: "progress", message: "Configuring region..." });
				let region = process.env.VERTEX_REGION;

				if (!region) {
					const regionChoice = await callbacks.onPrompt({
						message:
							"Select region:\n\n" +
							"  1. global (recommended for latest models)\n" +
							"  2. us-east5\n" +
							"  3. us-central1\n" +
							"  4. europe-west1\n" +
							"  5. asia-southeast1\n\n" +
							"Enter 1-5 or custom region name:",
					});

					const regionMap: Record<string, string> = {
						"1": "global",
						"2": "us-east5",
						"3": "us-central1",
						"4": "europe-west1",
						"5": "asia-southeast1",
					};
					region = regionMap[regionChoice || ""] || (regionChoice?.trim() ? validateRegion(regionChoice) : "global");
					process.env.VERTEX_REGION = region;
				}

				// Step 5: Enable Vertex AI API
				callbacks.onAuth({ type: "progress", message: "Checking Vertex AI API..." });
				try {
					if (!isVertexApiEnabled(gcloudPath)) {
						const enable = await callbacks.onPrompt({
							message: "Vertex AI API not enabled. Enable it now? (y/n)",
						});

						if (enable?.toLowerCase() === "y") {
							callbacks.onAuth({
								type: "progress",
								message: "Enabling Vertex AI API (this may take a minute)...",
							});
							execFileSync(gcloudPath, ["services", "enable", "aiplatform.googleapis.com"], {
								stdio: "inherit",
							});
							callbacks.onAuth({ type: "info", message: "Vertex AI API enabled!" });
						} else {
							callbacks.onAuth({
								type: "warning",
								message:
									"API not enabled. You'll need to enable it manually:\n" +
									`  gcloud services enable aiplatform.googleapis.com --project=${project}`,
							});
						}
					}
				} catch {
					callbacks.onAuth({
						type: "warning",
						message:
							"Could not check API status. If requests fail, enable it manually:\n" +
							`  gcloud services enable aiplatform.googleapis.com --project=${project}`,
					});
				}

				// Step 6: Test authentication
				callbacks.onAuth({ type: "progress", message: "Testing authentication..." });
				if (!checkGcloudAuthenticated(gcloudPath)) {
					throw new Error("Authentication test failed. Please run: gcloud auth login");
				}

				callbacks.onAuth({
					type: "success",
					message:
						`✓ Configured successfully!\n\n` +
						`Project: ${project}\n` +
						`Region: ${region}\n\n` +
						`Settings persisted to ~/.pi/agent/auth.json.\n` +
						`If authentication fails later, run: gcloud auth login`,
				});

				return {
					refresh: Date.now().toString(),
					access: "gcloud",
					expires: Date.now() + 1000 * 60 * 60 * 24 * 365,
					project,
					region,
				};
			},

			async refreshToken(credentials) {
				return { ...credentials, refresh: Date.now().toString() };
			},

			getApiKey(_credentials) {
				try {
					const gcloudPath = config.gcloudPath;
					const token = execFileSync(gcloudPath, ["auth", "print-access-token"], {
						encoding: "utf-8",
						timeout: 5000,
						stdio: ["ignore", "pipe", "pipe"],
					}).trim();

					if (!token || token.length < 20) {
						throw new Error("Invalid token from gcloud");
					}
					return token;
				} catch (error) {
					const msg = error instanceof Error ? error.message : "Unknown error";
					throw new Error(`Failed to get gcloud access token: ${msg}\n\nRun: gcloud auth login`);
				}
			},
		},

		models: [
			{
				id: "claude-opus-4-6",
				name: "Claude Opus 4.6 (Vertex)",
				reasoning: true,
				input: ["text", "image"],
				cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
				contextWindow: 200000,
				maxTokens: 64000,
			},
			{
				id: "claude-opus-4-5@20251101",
				name: "Claude Opus 4.5 (Vertex)",
				reasoning: true,
				input: ["text", "image"],
				cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
				contextWindow: 200000,
				maxTokens: 64000,
			},
			{
				id: "claude-sonnet-4-5@20250929",
				name: "Claude Sonnet 4.5 (Vertex)",
				reasoning: true,
				input: ["text", "image"],
				cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
				contextWindow: 200000,
				maxTokens: 64000,
			},
			{
				id: "claude-haiku-4-5@20251001",
				name: "Claude Haiku 4.5 (Vertex)",
				reasoning: true,
				input: ["text", "image"],
				cost: { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
				contextWindow: 200000,
				maxTokens: 64000,
			},
			{
				id: "claude-3-5-sonnet-v2@20241022",
				name: "Claude 3.5 Sonnet v2 (Vertex)",
				reasoning: false,
				input: ["text", "image"],
				cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
				contextWindow: 200000,
				maxTokens: 8192,
			},
			{
				id: "claude-3-5-sonnet@20240620",
				name: "Claude 3.5 Sonnet (Vertex)",
				reasoning: false,
				input: ["text", "image"],
				cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
				contextWindow: 200000,
				maxTokens: 8192,
			},
			{
				id: "claude-3-5-haiku@20241022",
				name: "Claude 3.5 Haiku (Vertex)",
				reasoning: false,
				input: ["text", "image"],
				cost: { input: 0.8, output: 4, cacheRead: 0.08, cacheWrite: 1 },
				contextWindow: 200000,
				maxTokens: 8192,
			},
			{
				id: "claude-3-opus@20240229",
				name: "Claude 3 Opus (Vertex)",
				reasoning: false,
				input: ["text", "image"],
				cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
				contextWindow: 200000,
				maxTokens: 4096,
			},
			{
				id: "claude-3-sonnet@20240229",
				name: "Claude 3 Sonnet (Vertex)",
				reasoning: false,
				input: ["text", "image"],
				cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
				contextWindow: 200000,
				maxTokens: 4096,
			},
			{
				id: "claude-3-haiku@20240307",
				name: "Claude 3 Haiku (Vertex)",
				reasoning: false,
				input: ["text", "image"],
				cost: { input: 0.25, output: 1.25, cacheRead: 0.03, cacheWrite: 0.3 },
				contextWindow: 200000,
				maxTokens: 4096,
			},
		],

		streamSimple: streamVertexAnthropic,
	});

	pi.on("session_start", async (_event, ctx) => {
		if (config.project === "your-gcp-project-id") {
			ctx.ui?.notify("Vertex AI: Run /login to configure project and region", "warning");
		}
	});
}
