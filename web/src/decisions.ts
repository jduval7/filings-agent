import type { Card, Cause, DecisionIn } from "./types";

/** What the reviewer has chosen on one card so far. */
export interface Draft {
  mode: "decide" | "escalate" | "research";
  cause?: Cause;
  keepOriginal: boolean; // false: use the restated (newest) figure, the usual choice
  note: string;
}

export const emptyDraft = (): Draft => ({ mode: "decide", keepOriginal: false, note: "" });

/** Why this draft can't be submitted yet, or null if it can. */
export function problem(card: Card, draft: Draft | undefined): string | null {
  if (!draft) return "not decided";
  if (draft.mode === "decide" && !draft.cause) return "pick the cause";
  if (draft.mode === "research" && (card.research_rounds_left ?? 0) <= 0)
    return "no research rounds left: decide or escalate";
  return null;
}

/** A warning (not a block) when the reviewer picks a cause code has ruled out. */
export function overridesRule(card: Card, draft: Draft | undefined): string | null {
  if (draft?.mode !== "decide" || !draft.cause) return null;
  const c = card.candidates.find((x) => x.cause === draft.cause);
  return c?.fit === "ruled_out" ? c.why : null;
}

export function toDecision(draft: Draft): DecisionIn {
  const note = draft.note.trim();
  const base = note ? { note } : {};
  if (draft.mode === "escalate") return { action: "escalate", ...base };
  if (draft.mode === "research") return { action: "research", ...base };
  return {
    action: draft.keepOriginal ? "use_original" : "accept",
    correct_cause: draft.cause,
    ...base,
  };
}

/** All decisions, or null while any card is incomplete (the API rejects partial reviews). */
export function toReview(
  cards: Card[],
  drafts: Record<string, Draft>,
): Record<string, DecisionIn> | null {
  if (cards.some((c) => problem(c, drafts[c.id]) !== null)) return null;
  return Object.fromEntries(cards.map((c) => [c.id, toDecision(drafts[c.id]!)]));
}
