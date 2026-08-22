import { describe, expect, it } from "vitest";
import { nearestNeighbors } from "../src/neighbors.js";

// 5 tokens, hidden=4. t0≈t1 (same direction), t2 orthogonal, t3 opposite,
// t4 is "untrained" (tiny norm, same direction as t0 — must be filtered out).
const wte = new Float32Array([
  1, 0, 0, 0,       // t0
  0.9, 0.1, 0, 0,   // t1
  0, 0, 1, 0,       // t2
  -1, 0, 0, 0,      // t3
  0.01, 0, 0, 0,    // t4 tiny norm
]);

describe("nearestNeighbors", () => {
  it("ranks by cosine similarity and excludes self", () => {
    const n = nearestNeighbors(wte, 4, 0, 3);
    expect(n[0].id).toBe(1);
    expect(n[0].sim).toBeGreaterThan(0.99);
    expect(n.map((x) => x.id)).not.toContain(0);
    expect(n[n.length - 1].id).toBe(3); // opposite direction ranks last
  });

  it("filters out untrained near-zero-norm tokens", () => {
    const n = nearestNeighbors(wte, 4, 0, 4);
    expect(n.map((x) => x.id)).not.toContain(4);
  });

  it("respects id range and handles out-of-range queries", () => {
    const n = nearestNeighbors(wte, 4, 0, 3, { minId: 2, maxId: 4 });
    expect(n.map((x) => x.id)).toEqual([2, 3]);
    expect(nearestNeighbors(wte, 4, 99, 3)).toEqual([]);
  });
});
