import type {
  RealtimeTranscriptionProviderConfig,
  RealtimeTranscriptionProviderPlugin,
  RealtimeTranscriptionSession,
  RealtimeTranscriptionSessionCreateRequest,
} from "openclaw/plugin-sdk/realtime-transcription";
import { normalizeResolvedSecretInputString } from "openclaw/plugin-sdk/secret-input";
import WebSocket from "ws";

type ElevenLabsScribeProviderConfig = {
  apiKey?: string;
  model?: string;
  languageCode?: string;
  vadSilenceThresholdSecs?: number;
  vadThreshold?: number;
  silenceDurationMs?: number;
  elevenlabsApiKey?: string;
  elevenlabsLanguageCode?: string;
};

type ElevenLabsScribeSessionConfig = RealtimeTranscriptionSessionCreateRequest & {
  apiKey: string;
  model: string;
  languageCode?: string;
  vadSilenceThresholdSecs: number;
  vadThreshold: number;
};

type ScribeEvent = {
  type?: string;
  event?: string;
  text?: string;
  error?: unknown;
  session_id?: string;
  is_final?: boolean;
};

function asFiniteNumber(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return value;
}

function trimToUndefined(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed || undefined;
}

function normalizeProviderConfig(
  config: RealtimeTranscriptionProviderConfig,
): ElevenLabsScribeProviderConfig {
  const raw = (config ?? {}) as Record<string, unknown>;
  return {
    apiKey:
      normalizeResolvedSecretInputString({
        value: raw.apiKey,
        path: "plugins.entries.voice-call.config.streaming.providers.elevenlabs-scribe.apiKey",
      }) ??
      normalizeResolvedSecretInputString({
        value: raw.elevenlabsApiKey,
        path: "plugins.entries.voice-call.config.streaming.providers.elevenlabs-scribe.elevenlabsApiKey",
      }),
    model: trimToUndefined(raw.model) ?? "scribe_v2_realtime",
    languageCode: trimToUndefined(raw.languageCode) ?? trimToUndefined(raw.elevenlabsLanguageCode),
    vadSilenceThresholdSecs:
      asFiniteNumber(raw.vadSilenceThresholdSecs) ??
      (asFiniteNumber(raw.silenceDurationMs) !== undefined
        ? (asFiniteNumber(raw.silenceDurationMs) as number) / 1000
        : undefined),
    vadThreshold: asFiniteNumber(raw.vadThreshold),
    silenceDurationMs: asFiniteNumber(raw.silenceDurationMs),
    elevenlabsApiKey: undefined,
    elevenlabsLanguageCode: undefined,
  };
}

const MULAW_DECODE_TABLE = new Int16Array(256);
(function buildMulawTable() {
  for (let i = 0; i < 256; i++) {
    const mu = ~i & 0xff;
    const sign = mu & 0x80;
    const exponent = (mu >> 4) & 0x07;
    const mantissa = mu & 0x0f;
    let sample = ((mantissa << 3) + 132) << exponent;
    sample -= 132;
    MULAW_DECODE_TABLE[i] = sign ? -sample : sample;
  }
})();

function mulawToPcm16k(mulaw: Buffer): Buffer {
  const inputSamples = mulaw.length;
  const pcm8k = new Int16Array(inputSamples);
  for (let i = 0; i < inputSamples; i++) {
    pcm8k[i] = MULAW_DECODE_TABLE[mulaw[i]];
  }

  const output = Buffer.alloc(inputSamples * 4);
  for (let i = 0; i < inputSamples; i++) {
    const s0 = pcm8k[i];
    const s1 = i + 1 < inputSamples ? pcm8k[i + 1] : s0;
    output.writeInt16LE(s0, i * 4);
    output.writeInt16LE(Math.round((s0 + s1) / 2), i * 4 + 2);
  }
  return output;
}

class ElevenLabsScribeTranscriptionSession implements RealtimeTranscriptionSession {
  private static readonly MAX_RECONNECT_ATTEMPTS = 5;
  private static readonly RECONNECT_DELAY_MS = 1000;
  private static readonly CONNECT_TIMEOUT_MS = 10_000;

  private ws: WebSocket | null = null;
  private connected = false;
  private closed = false;
  private reconnectAttempts = 0;

  constructor(private readonly config: ElevenLabsScribeSessionConfig) {}

  async connect(): Promise<void> {
    this.closed = false;
    this.reconnectAttempts = 0;
    await this.doConnect();
  }

  sendAudio(audio: Buffer): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      return;
    }
    const pcmAudio = mulawToPcm16k(audio);
    this.ws.send(
      JSON.stringify({
        user_audio_chunk: pcmAudio.toString("base64"),
      }),
    );
  }

  close(): void {
    this.closed = true;
    this.connected = false;
    if (this.ws) {
      this.ws.close(1000, "Transcription session closed");
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.connected;
  }

  private async doConnect(): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      const params = new URLSearchParams({
        model_id: this.config.model,
        commit_strategy: "vad",
        vad_silence_threshold_secs: String(this.config.vadSilenceThresholdSecs),
        vad_threshold: String(this.config.vadThreshold),
        audio_format: "pcm_16000",
      });
      if (this.config.languageCode) {
        params.set("language_code", this.config.languageCode);
      }

      const url = `wss://api.elevenlabs.io/v1/speech-to-text/realtime?${params.toString()}`;
      const ws = new WebSocket(url, {
        headers: {
          "xi-api-key": this.config.apiKey,
        },
      });
      this.ws = ws;

      const connectTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          try {
            ws.close();
          } catch {
            // ignore close errors during timeout cleanup
          }
          if (this.ws === ws) {
            this.ws = null;
          }
          reject(new Error("ElevenLabs Scribe STT connection timeout"));
        }
      }, ElevenLabsScribeTranscriptionSession.CONNECT_TIMEOUT_MS);

      ws.on("open", () => {
        clearTimeout(connectTimeout);
        this.connected = true;
        this.reconnectAttempts = 0;
        resolve();
      });

      ws.on("message", (data: Buffer) => {
        try {
          this.handleEvent(JSON.parse(data.toString()) as ScribeEvent);
        } catch (error) {
          this.config.onError?.(error instanceof Error ? error : new Error(String(error)));
        }
      });

      ws.on("error", (error) => {
        clearTimeout(connectTimeout);
        if (!this.connected) {
          reject(error);
          return;
        }
        this.config.onError?.(error instanceof Error ? error : new Error(String(error)));
      });

      ws.on("close", () => {
        clearTimeout(connectTimeout);
        this.connected = false;
        if (this.ws === ws) {
          this.ws = null;
        }
        if (this.closed) {
          return;
        }
        void this.attemptReconnect();
      });
    });
  }

  private async attemptReconnect(): Promise<void> {
    if (this.closed) {
      return;
    }
    if (this.reconnectAttempts >= ElevenLabsScribeTranscriptionSession.MAX_RECONNECT_ATTEMPTS) {
      this.config.onError?.(new Error("ElevenLabs Scribe reconnect limit reached"));
      return;
    }
    this.reconnectAttempts += 1;
    const delay =
      ElevenLabsScribeTranscriptionSession.RECONNECT_DELAY_MS *
      2 ** (this.reconnectAttempts - 1);
    await new Promise((resolve) => setTimeout(resolve, delay));
    if (this.closed) {
      return;
    }
    try {
      await this.doConnect();
    } catch (error) {
      this.config.onError?.(error instanceof Error ? error : new Error(String(error)));
      await this.attemptReconnect();
    }
  }

  private handleEvent(event: ScribeEvent): void {
    const type = event.type ?? event.event;
    switch (type) {
      case "speech_started":
        this.config.onSpeechStart?.();
        return;
      case "transcript.partial":
      case "partial_transcript":
      case "transcript":
        if (event.text && !event.is_final) {
          this.config.onPartial?.(event.text);
          return;
        }
        if (event.text) {
          this.config.onTranscript?.(event.text);
        }
        return;
      case "transcript.final":
      case "committed_transcript":
      case "final_transcript":
        if (event.text) {
          this.config.onTranscript?.(event.text);
        }
        return;
      case "error": {
        const detail =
          event.error instanceof Error
            ? event.error.message
            : typeof event.error === "string"
              ? event.error
              : JSON.stringify(event.error ?? "Unknown ElevenLabs Scribe error");
        this.config.onError?.(new Error(detail));
        return;
      }
      default:
        return;
    }
  }
}

export function buildElevenLabsScribeRealtimeTranscriptionProvider(): RealtimeTranscriptionProviderPlugin {
  return {
    id: "elevenlabs-scribe",
    label: "ElevenLabs Scribe Realtime Transcription",
    aliases: ["scribe", "elevenlabs"],
    autoSelectOrder: 20,
    resolveConfig: ({ rawConfig }) => normalizeProviderConfig(rawConfig),
    isConfigured: ({ providerConfig }) =>
      Boolean(normalizeProviderConfig(providerConfig).apiKey || process.env.ELEVENLABS_API_KEY),
    createSession: (req) => {
      const config = normalizeProviderConfig(req.providerConfig);
      const apiKey = config.apiKey || process.env.ELEVENLABS_API_KEY;
      if (!apiKey) {
        throw new Error("ElevenLabs API key missing");
      }
      return new ElevenLabsScribeTranscriptionSession({
        ...req,
        apiKey,
        model: config.model ?? "scribe_v2_realtime",
        languageCode: config.languageCode,
        vadSilenceThresholdSecs:
          config.vadSilenceThresholdSecs ??
          ((config.silenceDurationMs ?? 300) / 1000),
        vadThreshold: config.vadThreshold ?? 0.5,
      });
    },
  };
}
