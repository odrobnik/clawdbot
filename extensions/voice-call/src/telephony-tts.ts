import type { VoiceCallTtsConfig } from "./config.js";
import type { CoreConfig } from "./core-bridge.js";
import { convertPcmToMulaw8k, pcmToMulaw, resamplePcmTo8k } from "./telephony-audio.js";

export type TelephonyTtsRuntime = {
  textToSpeechTelephony: (params: {
    text: string;
    cfg: CoreConfig;
    prefsPath?: string;
  }) => Promise<{
    success: boolean;
    audioBuffer?: Buffer;
    sampleRate?: number;
    provider?: string;
    error?: string;
  }>;
};

export type PcmAudio = {
  audio: Buffer;
  sampleRate: number;
};

export type TelephonyTtsProvider = {
  synthesizeForTelephony: (text: string) => Promise<Buffer>;
  /**
   * Stream TTS audio as mu-law 8kHz chunks for real-time playback.
   * Falls back to non-streaming synthesis if direct streaming is unavailable.
   */
  streamForTelephony?: (
    text: string,
    signal?: AbortSignal,
  ) => AsyncGenerator<Buffer, void, unknown>;
  /** Synthesize raw PCM for web clients (without mu-law conversion). */
  synthesizeForWeb?: (text: string) => Promise<PcmAudio>;
  /** Stream raw PCM for web clients (without mu-law conversion). */
  streamForWeb?: (text: string, signal?: AbortSignal) => AsyncGenerator<PcmAudio, void, unknown>;
};

function buildElevenLabsBody(text: string, config: NonNullable<VoiceCallTtsConfig>): string {
  const elevenlabs = config.elevenlabs;
  const body: Record<string, unknown> = {
    text,
    model_id: elevenlabs?.modelId || "eleven_turbo_v2_5",
  };

  if (elevenlabs?.voiceSettings) {
    body.voice_settings = {
      stability: elevenlabs.voiceSettings.stability ?? 0.5,
      similarity_boost: elevenlabs.voiceSettings.similarityBoost ?? 0.75,
      ...(elevenlabs.voiceSettings.style != null && { style: elevenlabs.voiceSettings.style }),
      ...(elevenlabs.voiceSettings.useSpeakerBoost != null && {
        use_speaker_boost: elevenlabs.voiceSettings.useSpeakerBoost,
      }),
      ...(elevenlabs.voiceSettings.speed != null && { speed: elevenlabs.voiceSettings.speed }),
    };
  }
  if (elevenlabs?.languageCode) {
    body.language_code = elevenlabs.languageCode;
  }
  if (elevenlabs?.seed != null) {
    body.seed = elevenlabs.seed;
  }

  return JSON.stringify(body);
}

const PCM16_SAMPLE_BYTES = 2;

async function* streamFetchBody(
  response: Response,
  signal?: AbortSignal,
): AsyncGenerator<Buffer, void, unknown> {
  if (!response.body) {
    throw new Error("Streaming TTS returned no body");
  }

  const reader = response.body.getReader();
  const onAbort = () => {
    reader.cancel().catch(() => {});
  };
  signal?.addEventListener("abort", onAbort, { once: true });

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value && value.length > 0) {
        yield Buffer.from(value);
      }
    }
  } catch (err: unknown) {
    const isAbort = signal?.aborted || (err instanceof Error && err.name === "AbortError");
    if (!isAbort) {
      throw err;
    }
  } finally {
    signal?.removeEventListener("abort", onAbort);
    try {
      reader.releaseLock();
    } catch {
      // Reader may already be released after cancel
    }
  }
}

async function* streamFetchBodyAlignedPcm16(
  response: Response,
  signal?: AbortSignal,
): AsyncGenerator<Buffer, void, unknown> {
  // HTTP stream chunking is arbitrary and can split 16-bit PCM samples.
  // Preserve a trailing byte so downstream never receives odd-length PCM.
  let remainder = Buffer.alloc(0);

  for await (const chunk of streamFetchBody(response, signal)) {
    const combined = remainder.length > 0 ? Buffer.concat([remainder, chunk]) : chunk;
    const evenLength = combined.length - (combined.length % PCM16_SAMPLE_BYTES);
    if (evenLength > 0) {
      yield combined.subarray(0, evenLength);
    }
    remainder =
      evenLength < combined.length ? Buffer.from(combined.subarray(evenLength)) : Buffer.alloc(0);
  }
}

/**
 * Stream TTS directly from ElevenLabs in mu-law 8kHz format.
 */
async function* streamElevenLabsTelephony(
  text: string,
  config: NonNullable<VoiceCallTtsConfig>,
  signal?: AbortSignal,
): AsyncGenerator<Buffer, void, unknown> {
  const elevenlabs = config.elevenlabs;
  if (!elevenlabs?.apiKey || !elevenlabs?.voiceId) {
    throw new Error("ElevenLabs API key and voice ID required for streaming TTS");
  }

  const baseUrl = elevenlabs.baseUrl?.replace(/\/+$/, "") || "https://api.elevenlabs.io";
  const url = `${baseUrl}/v1/text-to-speech/${elevenlabs.voiceId}/stream?output_format=ulaw_8000`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "xi-api-key": elevenlabs.apiKey,
    },
    body: buildElevenLabsBody(text, config),
    signal,
  });

  const contentType = response.headers.get("content-type") ?? "unknown";
  console.log(
    `[voice-call] ElevenLabs streaming TTS response: ${response.status} ${response.statusText}; content-type=${contentType}`,
  );

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(`ElevenLabs streaming TTS failed: ${response.status} ${errorText}`);
  }

  yield* streamFetchBody(response, signal);
}

/**
 * Stream TTS from ElevenLabs as raw PCM16 @ 16kHz for web clients.
 */
async function* streamElevenLabsWebPcm(
  text: string,
  config: NonNullable<VoiceCallTtsConfig>,
  signal?: AbortSignal,
): AsyncGenerator<PcmAudio, void, unknown> {
  const elevenlabs = config.elevenlabs;
  if (!elevenlabs?.apiKey || !elevenlabs?.voiceId) {
    throw new Error("ElevenLabs API key and voice ID required for streaming TTS");
  }

  const baseUrl = elevenlabs.baseUrl?.replace(/\/+$/, "") || "https://api.elevenlabs.io";
  const url = `${baseUrl}/v1/text-to-speech/${elevenlabs.voiceId}/stream?output_format=pcm_16000`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "xi-api-key": elevenlabs.apiKey,
    },
    body: buildElevenLabsBody(text, config),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(`ElevenLabs PCM streaming TTS failed: ${response.status} ${errorText}`);
  }

  for await (const chunk of streamFetchBodyAlignedPcm16(response, signal)) {
    yield { audio: chunk, sampleRate: 16000 };
  }
}

/**
 * OpenAI TTS streaming: read PCM 24kHz and convert to mu-law 8kHz.
 */
async function* streamOpenAITelephony(
  text: string,
  config: NonNullable<VoiceCallTtsConfig>,
  signal?: AbortSignal,
): AsyncGenerator<Buffer, void, unknown> {
  const openai = config.openai;
  const apiKey = openai?.apiKey || process.env.OPENAI_API_KEY || "";
  if (!apiKey) {
    throw new Error("OpenAI API key required for streaming TTS");
  }

  const model = openai?.model || "gpt-4o-mini-tts";
  const voice = openai?.voice || "coral";

  const response = await fetch("https://api.openai.com/v1/audio/speech", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      input: text,
      voice,
      response_format: "pcm", // 24kHz PCM16
    }),
    signal,
  });

  console.log(
    `[voice-call] OpenAI streaming TTS response: ${response.status} ${response.statusText}`,
  );

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(`OpenAI streaming TTS failed: ${response.status} ${errorText}`);
  }

  // OpenAI PCM is 24kHz 16-bit mono. Convert to 8kHz mu-law in blocks.
  const BLOCK_SIZE = 4800; // 100ms of 24kHz PCM16 mono
  let remainder = Buffer.alloc(0);

  for await (const chunk of streamFetchBody(response, signal)) {
    const chunkBuf = Buffer.from(chunk);
    remainder = remainder.length > 0 ? Buffer.concat([remainder, chunkBuf]) : chunkBuf;

    while (remainder.length >= BLOCK_SIZE) {
      const blockBytes = Math.floor(BLOCK_SIZE / 2) * 2;
      const block = remainder.subarray(0, blockBytes);
      remainder = remainder.subarray(blockBytes);

      const pcm8k = resamplePcmTo8k(block, 24000);
      const mulaw = pcmToMulaw(pcm8k);
      if (mulaw.length > 0) {
        yield mulaw;
      }
    }
  }

  if (remainder.length >= 2) {
    const aligned = remainder.subarray(0, Math.floor(remainder.length / 2) * 2);
    const pcm8k = resamplePcmTo8k(aligned, 24000);
    const mulaw = pcmToMulaw(pcm8k);
    if (mulaw.length > 0) {
      yield mulaw;
    }
  }
}

/**
 * OpenAI TTS streaming as raw PCM16 @ 24kHz for web clients.
 */
async function* streamOpenAIWebPcm(
  text: string,
  config: NonNullable<VoiceCallTtsConfig>,
  signal?: AbortSignal,
): AsyncGenerator<PcmAudio, void, unknown> {
  const openai = config.openai;
  const apiKey = openai?.apiKey || process.env.OPENAI_API_KEY || "";
  if (!apiKey) {
    throw new Error("OpenAI API key required for streaming TTS");
  }

  const model = openai?.model || "gpt-4o-mini-tts";
  const voice = openai?.voice || "coral";

  const response = await fetch("https://api.openai.com/v1/audio/speech", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      input: text,
      voice,
      response_format: "pcm", // 24kHz PCM16
    }),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(`OpenAI PCM streaming TTS failed: ${response.status} ${errorText}`);
  }

  for await (const chunk of streamFetchBodyAlignedPcm16(response, signal)) {
    yield { audio: chunk, sampleRate: 24000 };
  }
}

export function createTelephonyTtsProvider(params: {
  coreConfig: CoreConfig;
  ttsOverride?: VoiceCallTtsConfig;
  runtime: TelephonyTtsRuntime;
}): TelephonyTtsProvider {
  const { coreConfig, ttsOverride, runtime } = params;

  // STRICT CONFIG SCOPE:
  // Voice calls must use ONLY plugins.entries.voice-call.config.tts.
  // We do NOT read or merge core messages.tts (messaging-channel TTS).
  if (!ttsOverride) {
    throw new Error(
      "voice-call TTS not configured: set plugins.entries.voice-call.config.tts (does not use messages.tts)",
    );
  }

  const effectiveConfig: CoreConfig = {
    ...coreConfig,
    messages: {
      ...coreConfig.messages,
      tts: ttsOverride,
    },
  };

  const ttsConfig = effectiveConfig.messages?.tts;
  const canStreamElevenLabs =
    ttsConfig?.provider === "elevenlabs" &&
    ttsConfig.elevenlabs?.apiKey &&
    ttsConfig.elevenlabs?.voiceId;

  const canStreamOpenAI =
    ttsConfig?.provider === "openai" && (ttsConfig.openai?.apiKey || process.env.OPENAI_API_KEY);

  return {
    synthesizeForTelephony: async (text: string) => {
      const result = await runtime.textToSpeechTelephony({
        text,
        cfg: effectiveConfig,
      });

      if (!result.success || !result.audioBuffer || !result.sampleRate) {
        throw new Error(result.error ?? "TTS conversion failed");
      }

      return convertPcmToMulaw8k(result.audioBuffer, result.sampleRate);
    },

    synthesizeForWeb: async (text: string) => {
      const result = await runtime.textToSpeechTelephony({
        text,
        cfg: effectiveConfig,
      });

      if (!result.success || !result.audioBuffer || !result.sampleRate) {
        throw new Error(result.error ?? "TTS conversion failed");
      }

      return { audio: result.audioBuffer, sampleRate: result.sampleRate };
    },

    // Streaming TTS: stream audio chunks as they arrive from the TTS provider
    ...((canStreamElevenLabs || canStreamOpenAI) && {
      streamForTelephony: (text: string, signal?: AbortSignal) =>
        canStreamElevenLabs
          ? streamElevenLabsTelephony(text, ttsConfig, signal)
          : streamOpenAITelephony(text, ttsConfig, signal),
      streamForWeb: (text: string, signal?: AbortSignal) =>
        canStreamElevenLabs
          ? streamElevenLabsWebPcm(text, ttsConfig, signal)
          : streamOpenAIWebPcm(text, ttsConfig, signal),
    }),
  };
}

// (dead code removed: deepMerge/isPlainObject were unused after TTS config refactor)
