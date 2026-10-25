/**
 * The UI's types against what the API really returns. The fixtures are generated from the real
 * graph by tests/support/api_fixtures.py; a Python test fails when they're stale.
 *
 * Two directions:
 *  - compile time: the key lists below must cover every field of the TypeScript types
 *    (a field added to a type without updating the list is a type error);
 *  - test time: every real response has exactly those keys (a field renamed or removed on the
 *    Python side fails here).
 */
import { describe, expect, it } from "vitest";
import awaiting from "./fixtures/analysis-awaiting.json";
import published from "./fixtures/analysis-published.json";
import researched from "./fixtures/analysis-researched.json";
import type { Analysis, Card, Candidate } from "./types";

const ANALYSIS_KEYS = [
  "thread_id", "status", "tickers", "branches", "branches_done", "facts", "rejected", "errors", "cost_usd", "paid_usd",
  "cached_branches", "discrepancies", "resolutions", "awaiting_review", "decisions",
  "research_rounds", "memo", "publication", "last_run_id", "runs", "review_token", "review_problems",
] as const satisfies readonly (keyof Analysis)[];
const CARD_KEYS = [
  "id", "change", "values", "changed_in", "candidates", "other_causes", "research_rounds",
  "research_rounds_left", "decide",
] as const satisfies readonly (keyof Card)[];
const CANDIDATE_KEYS = [
  "cause", "fit", "why", "evidence", "research", "research_not_shown",
] as const satisfies readonly (keyof Candidate)[];

// Compile-time completeness: these fail to type-check if a type gains a field the list lacks.
type Missing<T, K extends readonly unknown[]> = Exclude<keyof T, K[number]>;
const complete: [Missing<Analysis, typeof ANALYSIS_KEYS>, Missing<Card, typeof CARD_KEYS>, Missing<Candidate, typeof CANDIDATE_KEYS>] extends [never, never, never] ? true : never = true;

const sorted = (xs: readonly string[]) => [...xs].sort();

describe("API contract", () => {
  it("the key lists cover the types", () => expect(complete).toBe(true));

  for (const [name, fixture] of Object.entries({ awaiting, researched, published })) {
    it(`${name}: the response has exactly the fields the UI expects`, () => {
      expect(sorted(Object.keys(fixture))).toEqual(sorted(ANALYSIS_KEYS));
      for (const card of fixture.awaiting_review as unknown as Card[]) {
        expect(sorted(Object.keys(card))).toEqual(sorted(CARD_KEYS));
        for (const c of card.candidates) {
          // research fields appear only after a research round
          const required = CANDIDATE_KEYS.filter((k) => k !== "research" && k !== "research_not_shown");
          expect(Object.keys(c)).toEqual(expect.arrayContaining([...required]));
          expect(sorted(Object.keys(c)).every((k) => (CANDIDATE_KEYS as readonly string[]).includes(k))).toBe(true);
        }
      }
    });
  }
});
