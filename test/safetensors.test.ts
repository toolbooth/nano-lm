import { describe, expect, it } from "vitest";
import { f16ToF32, parseSafetensors, toArrayBuffer } from "../src/safetensors.js";

describe("f16 decode", () => {
  it("decodes known half-precision values", () => {
    expect(f16ToF32(0x3c00)).toBe(1);
    expect(f16ToF32(0xc000)).toBe(-2);
    expect(f16ToF32(0x0000)).toBe(0);
    expect(f16ToF32(0x3555)).toBeCloseTo(1 / 3, 3);
  });

  it("handles subnormals, infinities and NaN", () => {
    expect(f16ToF32(0x0001)).toBe(2 ** -24);
    expect(f16ToF32(0x8001)).toBe(-(2 ** -24));
    expect(f16ToF32(0x7c00)).toBe(Infinity);
    expect(f16ToF32(0xfc00)).toBe(-Infinity);
    expect(Number.isNaN(f16ToF32(0x7e00))).toBe(true);
    expect(f16ToF32(0x7bff)).toBe(65504); // largest finite half
  });
});

/** Build a safetensors file by hand: [u64 LE header len][JSON][F16 data]. */
function makeFile(
  tensors: Record<string, { shape: number[]; halves: number[] }>,
  opts: { dtype?: string; padHeaderTo?: number } = {}
): Uint8Array {
  const header: Record<string, unknown> = {};
  let offset = 0;
  const halves: number[] = [];
  for (const [name, t] of Object.entries(tensors)) {
    const n = t.halves.length * 2;
    header[name] = { dtype: opts.dtype ?? "F16", shape: t.shape, data_offsets: [offset, offset + n] };
    offset += n;
    halves.push(...t.halves);
  }
  header.__metadata__ = { format: "pt" };
  let json = JSON.stringify(header);
  if (opts.padHeaderTo !== undefined) json = json.padEnd(opts.padHeaderTo, " ");
  const jsonBytes = new TextEncoder().encode(json);
  const out = new Uint8Array(8 + jsonBytes.length + halves.length * 2);
  new DataView(out.buffer).setBigUint64(0, BigInt(jsonBytes.length), true);
  out.set(jsonBytes, 8);
  const view = new DataView(out.buffer, 8 + jsonBytes.length);
  halves.forEach((h, i) => view.setUint16(i * 2, h, true));
  return out;
}

const T = { a: { shape: [2, 2], halves: [0x3c00, 0xc000, 0x0000, 0x3555] }, b: { shape: [1], halves: [0x4000] } };

describe("parseSafetensors", () => {
  it("parses names, shapes and F16 data from an ArrayBuffer", () => {
    const file = makeFile(T, { padHeaderTo: 120 }); // even header → aligned data
    expect((8 + 120) % 2).toBe(0);
    const m = parseSafetensors(file.buffer as ArrayBuffer);
    expect([...m.keys()]).toEqual(["a", "b"]); // __metadata__ is skipped
    expect(m.get("a")!.shape).toEqual([2, 2]);
    expect(Array.from(m.get("a")!.data)).toEqual([1, -2, 0, f16ToF32(0x3555)]);
    expect(Array.from(m.get("b")!.data)).toEqual([2]);
  });

  it("decodes identically when the header length is odd (unaligned tensor data)", () => {
    const file = makeFile(T, { padHeaderTo: 121 });
    const m = parseSafetensors(file);
    expect(Array.from(m.get("a")!.data)).toEqual([1, -2, 0, f16ToF32(0x3555)]);
    expect(Array.from(m.get("b")!.data)).toEqual([2]);
  });

  it("accepts an offset Uint8Array view (pooled Node Buffers)", () => {
    const file = makeFile(T, { padHeaderTo: 120 });
    const padded = new Uint8Array(file.length + 3);
    padded.set(file, 3);
    const view = padded.subarray(3);
    expect(view.byteOffset).toBe(3);
    const m = parseSafetensors(view);
    expect(Array.from(m.get("a")!.data)).toEqual([1, -2, 0, f16ToF32(0x3555)]);
  });

  it("rejects dtypes other than F16", () => {
    const file = makeFile(T, { dtype: "F32", padHeaderTo: 120 });
    expect(() => parseSafetensors(file)).toThrow("unsupported dtype F32 for a");
  });
});

describe("toArrayBuffer", () => {
  it("returns whole-buffer views without copying and copies offset views", () => {
    const ab = new ArrayBuffer(8);
    expect(toArrayBuffer(ab)).toBe(ab);
    expect(toArrayBuffer(new Uint8Array(ab))).toBe(ab);
    const sub = new Uint8Array(ab, 2, 4);
    const copy = toArrayBuffer(sub);
    expect(copy).not.toBe(ab);
    expect(copy.byteLength).toBe(4);
  });
});
