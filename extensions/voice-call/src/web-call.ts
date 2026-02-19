/**
 * Web Call Handler
 *
 * Handles browser-based voice calls over WebSocket.
 * The browser sends PCM 16kHz 16-bit mono audio and receives the same format back.
 * Internally converts to/from mu-law 8kHz for the existing STT/TTS pipeline.
 */

import type { IncomingMessage } from "node:http";
import type { Duplex } from "node:stream";
import { WebSocket, WebSocketServer } from "ws";
import type { VoiceCallConfig } from "./config.js";
import type { CoreConfig } from "./core-bridge.js";
import type { MediaStreamConfig } from "./media-stream.js";
import { MediaStreamHandler } from "./media-stream.js";
import type { ElevenLabsScribeSTTProvider } from "./providers/stt-elevenlabs-scribe.js";
import type { OpenAIRealtimeSTTProvider } from "./providers/stt-openai-realtime.js";
import type { RealtimeSTTSession } from "./providers/stt-openai-realtime.js";
import { SilenceFiller } from "./silence-filler.js";
import { pcmToMulaw, resamplePcmTo8k } from "./telephony-audio.js";
import type { TelephonyTtsProvider } from "./telephony-tts.js";
import { createTelephonyTtsProvider } from "./telephony-tts.js";
import type { NormalizedEvent } from "./types.js";

type STTProvider = OpenAIRealtimeSTTProvider | ElevenLabsScribeSTTProvider;

/** Incoming messages from the browser */
type WebClientMessage = { type: "audio"; data: string } | { type: "hangup" };

/** Outgoing messages to the browser */
type WebServerMessage =
  | { type: "audio"; data: string }
  | { type: "transcript"; text: string; role: "user" | "agent" }
  | { type: "state"; value: "listening" | "thinking" | "speaking" }
  | { type: "ended" };

interface WebCallSession {
  id: string;
  ws: WebSocket;
  sttSession: RealtimeSTTSession;
  /** AbortController for current response generation */
  responseController: AbortController | null;
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
 * Convert mu-law 8kHz audio to PCM 16kHz 16-bit for browser playback.
 */
function mulawToLinear(mulaw: number): number {
  const BIAS = 33;
  let mu = ~mulaw & 0xff;
  const sign = mu & 0x80;
  const exponent = (mu >> 4) & 0x07;
  const mantissa = mu & 0x0f;
  let sample = ((mantissa << 3) + BIAS) << exponent;
  sample -= BIAS;
  return sign ? -sample : sample;
}

function mulaw8kToPcm16k(mulawBuf: Buffer): Buffer {
  // Decode mu-law to PCM 8kHz
  const pcm8k = Buffer.alloc(mulawBuf.length * 2);
  for (let i = 0; i < mulawBuf.length; i++) {
    const sample = mulawToLinear(mulawBuf[i]);
    pcm8k.writeInt16LE(sample, i * 2);
  }
  // Upsample 8kHz → 16kHz via linear interpolation
  const inputSamples = mulawBuf.length;
  const outputSamples = inputSamples * 2;
  const pcm16k = Buffer.alloc(outputSamples * 2);
  for (let i = 0; i < outputSamples; i++) {
    const srcPos = i / 2;
    const srcIdx = Math.floor(srcPos);
    const frac = srcPos - srcIdx;
    const s0 = pcm8k.readInt16LE(srcIdx * 2);
    const s1Idx = Math.min(srcIdx + 1, inputSamples - 1);
    const s1 = pcm8k.readInt16LE(s1Idx * 2);
    const sample = Math.round(s0 + frac * (s1 - s0));
    pcm16k.writeInt16LE(Math.max(-32768, Math.min(32767, sample)), i * 2);
  }
  return pcm16k;
}

/**
 * Convert PCM 16kHz 16-bit mono to mu-law 8kHz for the STT pipeline.
 */
function pcm16kToMulaw8k(pcmBuf: Buffer): Buffer {
  const pcm8k = resamplePcmTo8k(pcmBuf, 16000);
  return pcmToMulaw(pcm8k);
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

  constructor(deps: WebCallDeps) {
    this.deps = deps;
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
    console.log(`[web-call] New connection: ${sessionId}`);

    // Create STT session
    const sttSession = this.deps.sttProvider.createSession();

    const session: WebCallSession = {
      id: sessionId,
      ws,
      sttSession,
      responseController: null,
      closed: false,
    };

    this.sessions.set(sessionId, session);

    // Wire STT callbacks
    sttSession.onPartial((partial) => {
      sendMessage(ws, { type: "transcript", text: partial, role: "user" });
    });

    sttSession.onTranscript((transcript) => {
      console.log(`[web-call] Transcript for ${sessionId}: ${transcript}`);
      sendMessage(ws, { type: "transcript", text: transcript, role: "user" });
      sendMessage(ws, { type: "state", value: "thinking" });

      // Generate and speak response
      void this.handleResponse(session, transcript);
    });

    sttSession.onSpeechStart(() => {
      // Barge-in: cancel any in-progress response
      if (session.responseController) {
        session.responseController.abort();
        session.responseController = null;
      }
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
            if (session.closed) break;
            // Decode base64 PCM 16kHz → mu-law 8kHz for STT
            const pcmBuf = Buffer.from(msg.data, "base64");
            const mulawBuf = pcm16kToMulaw8k(pcmBuf);
            sttSession.sendAudio(mulawBuf);
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
    if (session.closed) return;

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

      if (controller.signal.aborted || session.closed) return;

      if (result.error) {
        console.error(`[web-call] Response generation error: ${result.error}`);
        sendMessage(session.ws, { type: "state", value: "listening" });
        return;
      }

      if (result.text) {
        console.log(`[web-call] AI response for ${session.id}: "${result.text}"`);
        sendMessage(session.ws, { type: "transcript", text: result.text, role: "agent" });
        sendMessage(session.ws, { type: "state", value: "speaking" });

        // Stream TTS audio to browser
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
   * Converts mu-law 8kHz from the TTS pipeline to PCM 16kHz for the browser.
   */
  private async streamTtsToClient(
    session: WebCallSession,
    text: string,
    signal: AbortSignal,
  ): Promise<void> {
    const ttsProvider = this.deps.ttsProvider;

    if (ttsProvider.streamForTelephony) {
      // Streaming mode: send chunks as they arrive
      for await (const mulawChunk of ttsProvider.streamForTelephony(text, signal)) {
        if (signal.aborted || session.closed) break;
        const pcm16k = mulaw8kToPcm16k(mulawChunk);
        sendMessage(session.ws, { type: "audio", data: pcm16k.toString("base64") });
      }
    } else {
      // Buffered mode: synthesize entire audio then send
      const mulawAudio = await ttsProvider.synthesizeForTelephony(text);
      if (signal.aborted || session.closed) return;
      // Send in chunks to avoid huge single messages
      const CHUNK_BYTES = 640; // 80ms at 8kHz mulaw
      for (let i = 0; i < mulawAudio.length; i += CHUNK_BYTES) {
        if (signal.aborted || session.closed) break;
        const chunk = mulawAudio.subarray(i, Math.min(i + CHUNK_BYTES, mulawAudio.length));
        const pcm16k = mulaw8kToPcm16k(chunk);
        sendMessage(session.ws, { type: "audio", data: pcm16k.toString("base64") });
      }
    }
  }

  /**
   * End a web call session and clean up.
   */
  private endSession(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (!session || session.closed) return;

    session.closed = true;
    console.log(`[web-call] Ending session: ${sessionId}`);

    // Cancel any in-progress response
    session.responseController?.abort();

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
