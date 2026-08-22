import { describe, expect, it } from "vitest";
import { geluNew, layerNorm, linear, softmaxRow } from "../src/ops.js";

describe("layerNorm", () => {
  it("normalizes each row to zero mean / unit variance, then scales and shifts", () => {
    const x = new Float32Array([1, 2, 3, 4, 10, 10, 10, 10]);
    const out = layerNorm(x, 2, 4, new Float32Array([1, 1, 1, 1]), new Float32Array(4), 0);
    // row 0: mean 2.5, var 1.25 → (x - 2.5) / sqrt(1.25)
    const s = Math.sqrt(1.25);
    expect(out[0]).toBeCloseTo(-1.5 / s, 5);
    expect(out[3]).toBeCloseTo(1.5 / s, 5);
    // row 1 is constant → zero (eps = 0 gives 0/0 = NaN guard is the caller's eps)
    const withEps = layerNorm(x, 2, 4, new Float32Array([2, 2, 2, 2]), new Float32Array([1, 1, 1, 1]), 1e-5);
    for (let d = 0; d < 4; d++) expect(withEps[4 + d]).toBeCloseTo(1, 5);
    expect(withEps[0]).toBeCloseTo(2 * (-1.5 / Math.sqrt(1.25 + 1e-5)) + 1, 5);
  });
});

describe("linear", () => {
  it("computes x·Wᵀ + b with W in [out, in] layout", () => {
    const x = new Float32Array([1, 2, 3, 4]); // seq 2, in 2
    const w = new Float32Array([1, 0, 0, 1, 1, 1]); // out 3: rows [1,0],[0,1],[1,1]
    const b = new Float32Array([10, 20, 30]);
    expect(Array.from(linear(x, 2, 2, 3, w, b))).toEqual([11, 22, 33, 13, 24, 37]);
    expect(Array.from(linear(x, 2, 2, 3, w, null))).toEqual([1, 2, 3, 3, 4, 7]);
  });
});

describe("geluNew", () => {
  it("matches the tanh approximation at known points", () => {
    const out = geluNew(new Float32Array([0, 1, -1, 3]));
    expect(out[0]).toBe(0);
    expect(out[1]).toBeCloseTo(0.841192, 5);
    expect(out[2]).toBeCloseTo(-0.158808, 5);
    expect(out[3]).toBeCloseTo(2.996363, 5);
  });
});

describe("softmaxRow", () => {
  it("normalizes a segment in place and leaves the rest untouched", () => {
    const arr = new Float32Array([9, 1, 1, 1, -Infinity, 9]);
    softmaxRow(arr, 1, 4);
    expect(arr[0]).toBe(9);
    expect(arr[5]).toBe(9);
    expect(arr[1] + arr[2] + arr[3] + arr[4]).toBeCloseTo(1, 6);
    expect(arr[1]).toBeCloseTo(1 / 3, 6);
    expect(arr[4]).toBe(0); // masked key
  });

  it("is stable for large scores (no overflow)", () => {
    const arr = new Float32Array([1000, 1000, 999]);
    softmaxRow(arr, 0, 3);
    expect(arr[0]).toBeCloseTo(arr[1], 6);
    expect(arr[0] + arr[1] + arr[2]).toBeCloseTo(1, 6);
  });
});
