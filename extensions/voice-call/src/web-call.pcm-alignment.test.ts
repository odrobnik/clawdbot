import { describe, expect, it } from "vitest";
import { chunkPcm16Even, Pcm16WebChunkAligner } from "./web-call.js";

function makeBytes(length: number): Buffer {
  return Buffer.from(Array.from({ length }, (_, i) => i % 256));
}

describe("PCM16 web chunk alignment", () => {
  it("keeps 100ms framing while preserving byte order across odd upstream chunks", () => {
    const source = makeBytes(3200 * 2 + 917);
    const chunkSizes = [101, 3333, 1, 777, 2048, 512, 545];
    const aligner = new Pcm16WebChunkAligner(3200);
    const emitted: Buffer[] = [];

    let offset = 0;
    let sizeIndex = 0;
    while (offset < source.length) {
      const size = chunkSizes[sizeIndex % chunkSizes.length] ?? 3200;
      sizeIndex += 1;
      const end = Math.min(offset + size, source.length);
      emitted.push(...aligner.push(source.subarray(offset, end)));
      offset = end;
    }

    const tail = aligner.flush();
    if (tail) {
      emitted.push(tail);
    }

    expect(emitted.every((chunk) => chunk.length % 2 === 0)).toBe(true);
    for (let i = 0; i < emitted.length - 1; i += 1) {
      expect(emitted[i]?.length).toBe(3200);
    }

    const merged = Buffer.concat(emitted);
    const expected = source.subarray(0, source.length - (source.length % 2));
    expect(merged.equals(expected)).toBe(true);
  });

  it("carries single-byte remainder into the next outgoing chunk", () => {
    const source = Buffer.from([0, 1, 2, 3, 4, 5, 6, 7]);
    const chunks = chunkPcm16Even(source, 3);

    expect(chunks.map((chunk) => Array.from(chunk.values()))).toEqual([
      [0, 1],
      [2, 3, 4, 5],
      [6, 7],
    ]);
    expect(chunks.every((chunk) => chunk.length % 2 === 0)).toBe(true);
  });
});
