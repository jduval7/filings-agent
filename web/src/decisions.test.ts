import { describe, expect, it } from "vitest";
import { emptyDraft, overridesRule, problem, toDecision, toReview, type Draft } from "./decisions";
import type { Card } from "./types";

const card = (over: Partial<Card> = {}): Card => ({
  id: "GE:net_income:FY2022",
  change: "net_income FY2022: 225 -> 339 (+50.7%)",
  values: [["A", "225"], ["B", "339"]],
  changed_in: ["B"],
  candidates: [
    { cause: "stock_split", fit: "ruled_out", why: "a stock split changes share counts, not dollar amounts", evidence: [] },
    { cause: "discontinued_operations", fit: "ruled_out", why: "a spin-off can't change total net income", evidence: [] },
    { cause: "accounting_change", fit: "possible", why: "no numeric check", evidence: ["we adopted ASU 2018-12"] },
    { cause: "reclassification", fit: "no_evidence", why: "not mentioned", evidence: [] },
  ],
  research_rounds_left: 2,
  ...over,
});

const draft = (over: Partial<Draft>): Draft => ({ ...emptyDraft(), ...over });

describe("decisions", () => {
  it("a decision needs a cause; escalation doesn't", () => {
    expect(problem(card(), undefined)).toBe("not decided");
    expect(problem(card(), draft({}))).toBe("pick the cause");
    expect(problem(card(), draft({ cause: "accounting_change" }))).toBeNull();
    expect(problem(card(), draft({ mode: "escalate" }))).toBeNull();
  });

  it("research needs rounds left", () => {
    expect(problem(card(), draft({ mode: "research" }))).toBeNull();
    expect(problem(card({ research_rounds_left: 0 }), draft({ mode: "research" }))).toMatch(/no research rounds/);
  });

  it("warns, without blocking, when the pick overrides a code rule", () => {
    expect(overridesRule(card(), draft({ cause: "discontinued_operations" }))).toMatch(/spin-off/);
    expect(overridesRule(card(), draft({ cause: "accounting_change" }))).toBeNull();
  });

  it("maps drafts to the API's decisions", () => {
    expect(toDecision(draft({ cause: "accounting_change", note: " ASU 2018-12 " }))).toEqual({
      action: "accept", correct_cause: "accounting_change", note: "ASU 2018-12",
    });
    expect(toDecision(draft({ cause: "accounting_change", keepOriginal: true }))).toEqual({
      action: "use_original", correct_cause: "accounting_change",
    });
    expect(toDecision(draft({ mode: "escalate", note: "none fit" }))).toEqual({ action: "escalate", note: "none fit" });
    expect(toDecision(draft({ mode: "research", note: "check the 8-K" }))).toEqual({ action: "research", note: "check the 8-K" });
  });

  it("submits only when every card is complete", () => {
    const a = card(), b = card({ id: "GE:revenue:FY2021" });
    expect(toReview([a, b], { [a.id]: draft({ cause: "accounting_change" }) })).toBeNull();
    const all = toReview([a, b], { [a.id]: draft({ cause: "accounting_change" }), [b.id]: draft({ mode: "escalate" }) });
    expect(all).toEqual({ [a.id]: { action: "accept", correct_cause: "accounting_change" }, [b.id]: { action: "escalate" } });
  });
});
