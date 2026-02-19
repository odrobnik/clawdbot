/**
 * Web Call Handler
 *
 * Handles browser-based voice calls over WebSocket.
 * Browser I/O is PCM16@16kHz base64. Internally we can run either:
 * - ulaw8k (legacy): convert browser PCM -> mu-law for STT/TTS pipeline
 * - pcm16_16k (fullband): keep PCM end-to-end for web calls
 */

import fs from "node:fs";
import type { IncomingMessage } from "node:http";
import path from "node:path";
import type { Duplex } from "node:stream";
import { fileURLToPath } from "node:url";
import { WebSocket, WebSocketServer } from "ws";
import type { VoiceCallConfig } from "./config.js";
import type { CoreConfig } from "./core-bridge.js";
import type { ElevenLabsScribeSTTProvider } from "./providers/stt-elevenlabs-scribe.js";
import type {
  OpenAIRealtimeSTTProvider,
  RealtimeSTTSession,
  RealtimeSTTSessionOptions,
} from "./providers/stt-openai-realtime.js";
import {
  mulaw8kToPcm16k,
  pcmToMulaw,
  resamplePcmTo16k,
  resamplePcmTo8k,
} from "./telephony-audio.js";
import type { TelephonyTtsProvider } from "./telephony-tts.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ASSETS_DIR = path.resolve(__dirname, "..", "assets");

type STTProvider = OpenAIRealtimeSTTProvider | ElevenLabsScribeSTTProvider;

type WebAudioFormat = "ulaw8k" | "pcm16_16k";

/** Incoming messages from the browser */
type WebClientMessage = { type: "audio"; data: string } | { type: "hangup" };

/** Outgoing messages to the browser */
type WebServerMessage =
  | { type: "audio"; data: string }
  | { type: "audio_clear" }
  | { type: "transcript"; text: string; role: "user" | "agent"; final: boolean }
  | { type: "state"; value: "listening" | "thinking" | "speaking" }
  | { type: "ended" };

interface WebCallSession {
  id: string;
  ws: WebSocket;
  sttSession: RealtimeSTTSession;
  /** AbortController for current response generation */
  responseController: AbortController | null;
  fillerTimer: ReturnType<typeof setTimeout> | null;
  fillerController: AbortController | null;
  closed: boolean;
}

/**
 * Send a typed JSON message to the browser.
 */
function sendMessage(ws: WebSocket, msg: WebServerMessage): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

/**
 * Convert PCM 16kHz 16-bit mono to mu-law 8kHz for the legacy STT pipeline.
 */
function pcm16kToMulaw8k(pcmBuf: Buffer): Buffer {
  const pcm8k = resamplePcmTo8k(pcmBuf, 16000);
  return pcmToMulaw(pcm8k);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const PCM16_WEB_CHUNK_BYTES = 3200; // 100ms @ 16kHz PCM16 mono
const PCM16_SAMPLE_BYTES = 2;

/**
 * PCM16 must stay aligned on 2-byte boundaries. If a websocket "audio" message
 * ever contains an odd number of bytes, browser-side Int16 decoding shifts by
 * one byte and all subsequent playback sounds like static.
 */
export class Pcm16WebChunkAligner {
  private frameRemainder = Buffer.alloc(0);
  private trailingByteRemainder = Buffer.alloc(0); // 0 or 1 byte

  constructor(private readonly chunkBytes = PCM16_WEB_CHUNK_BYTES) {}

  push(pcmChunk: Buffer): Buffer[] {
    if (pcmChunk.length === 0) {
      return [];
    }

    let combined = pcmChunk;
    if (this.frameRemainder.length > 0 || this.trailingByteRemainder.length > 0) {
      combined = Buffer.concat([this.frameRemainder, this.trailingByteRemainder, pcmChunk]);
      this.frameRemainder = Buffer.alloc(0);
      this.trailingByteRemainder = Buffer.alloc(0);
    }

    const evenLength = combined.length - (combined.length % PCM16_SAMPLE_BYTES);
    const aligned = combined.subarray(0, evenLength);
    if (evenLength < combined.length) {
      this.trailingByteRemainder = Buffer.from(combined.subarray(evenLength));
    }

    const frames: Buffer[] = [];
    for (let i = 0; i + this.chunkBytes <= aligned.length; i += this.chunkBytes) {
      frames.push(Buffer.from(aligned.subarray(i, i + this.chunkBytes)));
    }

    const sentBytes = Math.floor(aligned.length / this.chunkBytes) * this.chunkBytes;
    if (sentBytes < aligned.length) {
      this.frameRemainder = Buffer.from(aligned.subarray(sentBytes));
    }

    return frames;
  }

  flush(): Buffer | null {
    const buffered =
      this.trailingByteRemainder.length > 0
        ? Buffer.concat([this.frameRemainder, this.trailingByteRemainder])
        : this.frameRemainder;
    const evenLength = buffered.length - (buffered.length % PCM16_SAMPLE_BYTES);
    const tail = evenLength > 0 ? Buffer.from(buffered.subarray(0, evenLength)) : null;
    this.reset();
    return tail;
  }

  reset(): void {
    this.frameRemainder = Buffer.alloc(0);
    this.trailingByteRemainder = Buffer.alloc(0);
  }
}

export function chunkPcm16Even(pcmAudio: Buffer, chunkBytes = PCM16_WEB_CHUNK_BYTES): Buffer[] {
  const chunks: Buffer[] = [];
  let carry = Buffer.alloc(0);

  for (let i = 0; i < pcmAudio.length; i += chunkBytes) {
    const slice = pcmAudio.subarray(i, Math.min(i + chunkBytes, pcmAudio.length));
    let merged = slice;

    if (carry.length > 0) {
      merged = Buffer.concat([carry, slice]);
      carry = Buffer.alloc(0);
    }

    const evenLength = merged.length - (merged.length % PCM16_SAMPLE_BYTES);
    if (evenLength > 0) {
      chunks.push(Buffer.from(merged.subarray(0, evenLength)));
    }
    if (evenLength < merged.length) {
      carry = Buffer.from(merged.subarray(evenLength));
    }
  }

  return chunks;
}

// Pre-loaded web filler clips (converted to PCM16/16k)
let webFillerCache: Map<string, Buffer> | null = null;

function attenuateMulaw(buf: Buffer, factor: number): Buffer {
  if (factor <= 1) {
    return buf;
  }
  const out = Buffer.alloc(buf.length);
  const silence = 0xff;
  for (let i = 0; i < buf.length; i++) {
    out[i] = Math.round(buf[i] + (silence - buf[i]) * (1 - 1 / factor));
  }
  return out;
}

function loadWebFillerClips(volumeReduction: number): Map<string, Buffer> {
  if (webFillerCache) {
    return webFillerCache;
  }

  webFillerCache = new Map();
  for (const clip of ["typing", "processing"] as const) {
    const clipPath = path.join(ASSETS_DIR, `${clip}.raw`);
    if (!fs.existsSync(clipPath)) {
      continue;
    }

    const mulawRaw = fs.readFileSync(clipPath);
    const attenuated = attenuateMulaw(mulawRaw, volumeReduction);
    const pcm16k = mulaw8kToPcm16k(attenuated);
    webFillerCache.set(clip, pcm16k);
  }

  return webFillerCache;
}

export type WebCallDeps = {
  config: VoiceCallConfig;
  coreConfig: CoreConfig;
  sttProvider: STTProvider;
  ttsProvider: TelephonyTtsProvider;
  onAutoRespond: (sessionId: string, transcript: string) => Promise<void>;
};

/**
 * WebSocket handler for browser-based web phone calls.
 */
export class WebCallHandler {
  private wss: WebSocketServer | null = null;
  private sessions = new Map<string, WebCallSession>();
  private deps: WebCallDeps;
  private readonly audioFormat: WebAudioFormat;
  private readonly fillerEnabled: boolean;
  private readonly fillerThresholdMs: number;
  private readonly fillerSfxSet: "typing" | "processing";

  constructor(deps: WebCallDeps) {
    this.deps = deps;
    this.audioFormat = deps.config.web?.audioFormat ?? "ulaw8k";
    this.fillerEnabled = deps.config.silenceFiller?.enabled ?? true;
    this.fillerThresholdMs = deps.config.silenceFiller?.thresholdMs ?? 3500;
    this.fillerSfxSet = deps.config.silenceFiller?.sfxSet ?? "typing";
  }

  /**
   * Handle WebSocket upgrade for web call connections.
   */
  handleUpgrade(request: IncomingMessage, socket: Duplex, head: Buffer): void {
    // Authenticate via query param
    const url = new URL(request.url || "/", `http://${request.headers.host}`);
    const token = url.searchParams.get("token");
    const expectedToken = this.deps.config.web?.token;

    if (!expectedToken || token !== expectedToken) {
      console.warn("[web-call] Rejected connection: invalid or missing token");
      socket.write("HTTP/1.1 401 Unauthorized\r\n\r\n");
      socket.destroy();
      return;
    }

    if (!this.wss) {
      this.wss = new WebSocketServer({ noServer: true });
      this.wss.on("connection", (ws) => this.handleConnection(ws));
    }

    this.wss.handleUpgrade(request, socket, head, (ws) => {
      this.wss?.emit("connection", ws, request);
    });
  }

  /**
   * Handle a new WebSocket connection from a browser client.
   */
  private async handleConnection(ws: WebSocket): Promise<void> {
    const sessionId = `web-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    console.log(`[web-call] New connection: ${sessionId} (audioFormat=${this.audioFormat})`);

    const sttSessionOptions: RealtimeSTTSessionOptions =
      this.audioFormat === "pcm16_16k"
        ? { inputAudioFormat: "pcm16", inputSampleRate: 16000 }
        : { inputAudioFormat: "g711_ulaw", inputSampleRate: 8000 };

    // Create STT session
    const sttSession = this.deps.sttProvider.createSession(sttSessionOptions);

    const session: WebCallSession = {
      id: sessionId,
      ws,
      sttSession,
      responseController: null,
      fillerTimer: null,
      fillerController: null,
      closed: false,
    };

    this.sessions.set(sessionId, session);

    // Wire STT callbacks
    sttSession.onPartial((partial) => {
      sendMessage(ws, { type: "transcript", text: partial, role: "user", final: false });
    });

    sttSession.onTranscript((transcript) => {
      console.log(`[web-call] Transcript for ${sessionId}: ${transcript}`);
      sendMessage(ws, { type: "transcript", text: transcript, role: "user", final: true });
      sendMessage(ws, { type: "state", value: "thinking" });
      this.startFiller(session);

      // Generate and speak response
      void this.handleResponse(session, transcript);
    });

    sttSession.onSpeechStart(() => {
      // Barge-in: cancel any in-progress response and clear queued audio
      if (session.responseController) {
        session.responseController.abort();
        session.responseController = null;
      }
      this.stopFiller(session, true);
      sendMessage(ws, { type: "state", value: "listening" });
    });

    // Connect STT (non-blocking)
    sttSession.connect().catch((err) => {
      console.warn(`[web-call] STT connection failed for ${sessionId}:`, err.message);
    });

    sendMessage(ws, { type: "state", value: "listening" });

    // Handle incoming messages
    ws.on("message", (data: Buffer) => {
      try {
        const msg = JSON.parse(data.toString()) as WebClientMessage;

        switch (msg.type) {
          case "audio": {
            if (session.closed) {
              break;
            }

            const pcmBuf = Buffer.from(msg.data, "base64");
            if (this.audioFormat === "pcm16_16k") {
              sttSession.sendAudio(pcmBuf);
            } else {
              // Legacy pipeline: browser PCM16k -> mu-law8k for STT
              const mulawBuf = pcm16kToMulaw8k(pcmBuf);
              sttSession.sendAudio(mulawBuf);
            }
            break;
          }
          case "hangup":
            this.endSession(sessionId);
            break;
        }
      } catch (err) {
        console.error(`[web-call] Error processing message from ${sessionId}:`, err);
      }
    });

    ws.on("close", () => {
      this.endSession(sessionId);
    });

    ws.on("error", (err) => {
      console.error(`[web-call] WebSocket error for ${sessionId}:`, err);
    });
  }

  /**
   * Handle agent response generation and TTS playback to the browser.
   */
  private async handleResponse(session: WebCallSession, userMessage: string): Promise<void> {
    if (session.closed) {
      return;
    }

    const controller = new AbortController();
    session.responseController = controller;

    try {
      const { generateVoiceResponse } = await import("./response-generator.js");

      const result = await generateVoiceResponse({
        voiceConfig: this.deps.config,
        coreConfig: this.deps.coreConfig,
        callId: session.id,
        from: "web-client",
        transcript: [], // TODO: maintain conversation history per session
        userMessage,
      });

      if (controller.signal.aborted || session.closed) {
        return;
      }

      if (result.error) {
        console.error(`[web-call] Response generation error: ${result.error}`);
        this.stopFiller(session, false);
        sendMessage(session.ws, { type: "state", value: "listening" });
        return;
      }

      if (result.text) {
        console.log(`[web-call] AI response for ${session.id}: "${result.text}"`);
        sendMessage(session.ws, {
          type: "transcript",
          text: result.text,
          role: "agent",
          final: true,
        });

        // Stop filler and clear any queued filler chunks before speech starts.
        this.stopFiller(session, true);
        sendMessage(session.ws, { type: "state", value: "speaking" });

        await this.streamTtsToClient(session, result.text, controller.signal);

        if (result.endCall) {
          console.log(`[web-call] Agent requested end call for ${session.id}`);
          setTimeout(() => this.endSession(session.id), 500);
          return;
        }
      }

      if (!controller.signal.aborted && !session.closed) {
        sendMessage(session.ws, { type: "state", value: "listening" });
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        console.error(`[web-call] Response error for ${session.id}:`, err);
      }
      this.stopFiller(session, false);
      if (!session.closed) {
        sendMessage(session.ws, { type: "state", value: "listening" });
      }
    } finally {
      if (session.responseController === controller) {
        session.responseController = null;
      }
    }
  }

  /**
   * Stream TTS audio to the browser client.
   */
  private async streamTtsToClient(
    session: WebCallSession,
    text: string,
    signal: AbortSignal,
  ): Promise<void> {
    const ttsProvider = this.deps.ttsProvider;

    if (this.audioFormat === "pcm16_16k") {
      if (ttsProvider.streamForWeb) {
        const aligner = new Pcm16WebChunkAligner(PCM16_WEB_CHUNK_BYTES);
        try {
          for await (const chunk of ttsProvider.streamForWeb(text, signal)) {
            if (signal.aborted || session.closed) {
              break;
            }
            const pcm16k =
              chunk.sampleRate === 16000
                ? chunk.audio
                : resamplePcmTo16k(chunk.audio, chunk.sampleRate);

            const frames = aligner.push(pcm16k);
            for (const frame of frames) {
              if (signal.aborted || session.closed) {
                break;
              }
              await this.sendPcmToClient(session, frame, signal);
            }
          }

          if (!signal.aborted && !session.closed) {
            const tail = aligner.flush();
            if (tail && tail.length > 0) {
              await this.sendPcmToClient(session, tail, signal);
            }
          }
        } finally {
          // Abort/stop should not leak remainders into a future response.
          aligner.reset();
        }
        return;
      }

      if (ttsProvider.synthesizeForWeb) {
        const pcm = await ttsProvider.synthesizeForWeb(text);
        if (signal.aborted || session.closed) {
          return;
        }
        const pcm16k =
          pcm.sampleRate === 16000 ? pcm.audio : resamplePcmTo16k(pcm.audio, pcm.sampleRate);
        await this.sendPcmToClient(session, pcm16k, signal);
        return;
      }
    }

    // Legacy compatibility path: telephony mu-law -> browser PCM16k
    if (ttsProvider.streamForTelephony) {
      for await (const mulawChunk of ttsProvider.streamForTelephony(text, signal)) {
        if (signal.aborted || session.closed) {
          break;
        }
        const pcm16k = mulaw8kToPcm16k(mulawChunk);
        sendMessage(session.ws, { type: "audio", data: pcm16k.toString("base64") });
      }
    } else {
      const mulawAudio = await ttsProvider.synthesizeForTelephony(text);
      if (signal.aborted || session.closed) {
        return;
      }
      const CHUNK_BYTES = 640; // 80ms at 8kHz mulaw
      for (let i = 0; i < mulawAudio.length; i += CHUNK_BYTES) {
        if (signal.aborted || session.closed) {
          break;
        }
        const chunk = mulawAudio.subarray(i, Math.min(i + CHUNK_BYTES, mulawAudio.length));
        const pcm16k = mulaw8kToPcm16k(chunk);
        sendMessage(session.ws, { type: "audio", data: pcm16k.toString("base64") });
      }
    }
  }

  private async sendPcmToClient(
    session: WebCallSession,
    pcmAudio: Buffer,
    signal: AbortSignal,
  ): Promise<void> {
    // Never emit odd-length chunks: Int16Array decoding in the browser requires
    // exact 2-byte sample alignment.
    const chunks = chunkPcm16Even(pcmAudio, PCM16_WEB_CHUNK_BYTES);
    for (const chunk of chunks) {
      if (signal.aborted || session.closed) {
        break;
      }
      sendMessage(session.ws, { type: "audio", data: chunk.toString("base64") });
      await sleep(100);
    }
  }

  private startFiller(session: WebCallSession): void {
    if (!this.fillerEnabled || session.closed) {
      return;
    }

    this.stopFiller(session, false);

    session.fillerTimer = setTimeout(() => {
      session.fillerTimer = null;
      if (session.closed) {
        return;
      }

      const controller = new AbortController();
      session.fillerController = controller;
      void this.playFiller(session, controller.signal).finally(() => {
        if (session.fillerController === controller) {
          session.fillerController = null;
        }
      });
    }, this.fillerThresholdMs);
  }

  private stopFiller(session: WebCallSession, clearAudio: boolean): void {
    if (session.fillerTimer) {
      clearTimeout(session.fillerTimer);
      session.fillerTimer = null;
    }
    if (session.fillerController) {
      session.fillerController.abort();
      session.fillerController = null;
    }
    if (clearAudio) {
      sendMessage(session.ws, { type: "audio_clear" });
    }
  }

  private async playFiller(session: WebCallSession, signal: AbortSignal): Promise<void> {
    const clips = loadWebFillerClips(2);
    const clipNames = this.fillerSfxSet === "processing" ? ["processing"] : ["typing"];
    try {
      while (!signal.aborted && !session.closed) {
        const name = clipNames[Math.floor(Math.random() * clipNames.length)];
        const clip = name ? clips.get(name) : undefined;
        if (!clip) {
          return;
        }

        for (
          let i = 0;
          i < clip.length && !signal.aborted && !session.closed;
          i += PCM16_WEB_CHUNK_BYTES
        ) {
          const chunk = clip.subarray(i, Math.min(i + PCM16_WEB_CHUNK_BYTES, clip.length));
          sendMessage(session.ws, { type: "audio", data: chunk.toString("base64") });
          await sleep(100);
        }

        if (!signal.aborted) {
          await sleep(500);
        }
      }
    } catch {
      // aborted/closed
    }
  }

  /**
   * End a web call session and clean up.
   */
  private endSession(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (!session || session.closed) {
      return;
    }

    session.closed = true;
    console.log(`[web-call] Ending session: ${sessionId}`);

    // Cancel any in-progress response
    session.responseController?.abort();
    this.stopFiller(session, false);

    // Close STT
    session.sttSession.close();

    // Notify browser
    sendMessage(session.ws, { type: "ended" });

    // Close WebSocket
    if (session.ws.readyState === WebSocket.OPEN) {
      session.ws.close(1000, "Call ended");
    }

    this.sessions.delete(sessionId);
  }

  /**
   * Close all active sessions.
   */
  closeAll(): void {
    for (const sessionId of this.sessions.keys()) {
      this.endSession(sessionId);
    }
  }
}
