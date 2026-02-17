import type { VoiceCallConfig } from "./config.js";
import type { CoreAgentDeps, CoreConfig } from "./core-bridge.js";

function resolveModelPrimary(raw: unknown): string | undefined {
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    return trimmed || undefined;
  }
  if (!raw || typeof raw !== "object") {
    return undefined;
  }
  const primary = (raw as { primary?: unknown }).primary;
  if (typeof primary !== "string") {
    return undefined;
  }
  const trimmed = primary.trim();
  return trimmed || undefined;
}

function resolveVoiceAgentPrimaryModel(coreConfig: CoreConfig | undefined, agentId: string): string | undefined {
  const agents =
    (coreConfig?.agents as { defaults?: { model?: unknown }; list?: unknown } | undefined) ??
    undefined;
  const entries = Array.isArray(agents?.list) ? agents.list : [];
  const normalizedAgentId = agentId.trim().toLowerCase();
  const selectedAgent = entries.find((entry) => {
    if (!entry || typeof entry !== "object") {
      return false;
    }
    const id = (entry as { id?: unknown }).id;
    return typeof id === "string" && id.trim().toLowerCase() === normalizedAgentId;
  }) as { model?: unknown } | undefined;

  return resolveModelPrimary(selectedAgent?.model) ?? resolveModelPrimary(agents?.defaults?.model);
}

export function resolveVoiceResponseModel(params: {
  voiceConfig: VoiceCallConfig;
  coreConfig?: CoreConfig;
  agentId?: string;
  agentRuntime: CoreAgentDeps;
}): {
  modelRef: string;
  provider: string;
  model: string;
} {
  const modelRef =
    params.voiceConfig.responseModel ??
    resolveVoiceAgentPrimaryModel(params.coreConfig, params.agentId ?? "main") ??
    `${params.agentRuntime.defaults.provider}/${params.agentRuntime.defaults.model}`;
  const slashIndex = modelRef.indexOf("/");

  return {
    modelRef,
    provider:
      slashIndex === -1 ? params.agentRuntime.defaults.provider : modelRef.slice(0, slashIndex),
    model: slashIndex === -1 ? modelRef : modelRef.slice(slashIndex + 1),
  };
}
