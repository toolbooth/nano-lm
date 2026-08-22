import { describe, expect, it } from "vitest";
import { diagnoseHeads } from "../src/diagnose.js";

describe("diagnoseHeads", () => {
  it("finds the previous-token head and the anchor head", () => {
    const seq = 3;
    // head A: q1→k0, q2→k1 (always the previous token)
    const a = new Float32Array([1, 0, 0, 1, 0, 0, 0, 1, 0]);
    // head B: every query goes back to token 0
    const b = new Float32Array([1, 0, 0, 1, 0, 0, 1, 0, 0]);
    const d = diagnoseHeads([[a, b]], seq)!;
    expect(d.prevToken).toMatchObject({ layer: 0, head: 0 });
    expect(d.prevToken.score).toBeCloseTo(1, 5);
    expect(d.firstToken).toMatchObject({ layer: 0, head: 1 });
    expect(d.diffuse.layer).toBe(0);
  });

  it("returns null for tiny sequences", () => {
    expect(diagnoseHeads([[new Float32Array(4)]], 2)).toBeNull();
  });
});
