/** Properties of the decision logic for any set of cards and drafts (fast-check). */
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { problem, toDecision, toReview, type Draft } from "./decisions";
import { formatAmount } from "./format";
import { CAUSES, type Card } from "./types";

const card = (id: string, roundsLeft: number): Card => ({
  id, change: "", values: [["A", "1"], ["B", "2"]], changed_in: ["B"],
  candidates: CAUSES.map((cause) => ({ cause, fit: "possible", why: "", evidence: [] })),
  research_rounds_left: roundsLeft,
});

const draft: fc.Arbitrary<Draft | undefined> = fc.option(
  fc.record({
    mode: fc.constantFrom("decide", "escalate", "research"),
    cause: fc.option(fc.constantFrom(...CAUSES), { nil: undefined }),
    keepOriginal: fc.boolean(),
    note: fc.string(),
  }) as fc.Arbitrary<Draft>,
  { nil: undefined },
);

describe("decision properties", () => {
  it("a review is produced exactly when every card is complete, with one decision per card", () => {
    fc.assert(
      fc.property(fc.array(fc.tuple(fc.integer({ min: 0, max: 2 }), draft), { maxLength: 8 }), (items) => {
        const cards = items.map(([left], i) => card(`C${i}`, left));
        const drafts = Object.fromEntries(items.flatMap(([, d], i) => (d ? [[`C${i}`, d]] : [])));
        const review = toReview(cards, drafts);
        const complete = cards.every((c) => problem(c, drafts[c.id]) === null);
        expect(review !== null).toBe(complete);
        if (review) expect(Object.keys(review).sort()).toEqual(cards.map((c) => c.id).sort());
      }),
    );
  });

  it("only a decision carries a cause, and it's the one picked", () => {
    fc.assert(
      fc.property(draft, (d) => {
        if (!d || (d.mode === "decide" && !d.cause)) return;
        const out = toDecision(d);
        if (d.mode === "decide") expect(out.correct_cause).toBe(d.cause);
        else expect(out.correct_cause).toBeUndefined();
        expect(out.note ?? "").toBe(d.note.trim());
      }),
    );
  });

  it("amounts keep their sign and unit", () => {
    fc.assert(
      fc.property(fc.integer({ min: -1e13, max: 1e13 }), fc.constantFrom("revenue", "diluted_shares"), (n, metric) => {
        const text = formatAmount(String(n), metric);
        expect(text.startsWith("-")).toBe(n < 0);
        expect(text.endsWith("shares")).toBe(metric === "diluted_shares");
        expect(text.includes("$")).toBe(metric !== "diluted_shares");
      }),
    );
  });
});
