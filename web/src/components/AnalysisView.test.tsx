import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import published from "../fixtures/analysis-published.json";
import researched from "../fixtures/analysis-researched.json";
import type { Analysis, Card } from "../types";
import { AnalysisView, humanError, step } from "./AnalysisView";
import { START } from "./Progress";

const card = (id: string): Card => ({
  id,
  change: `${id}: 100 -> 60 (-40.0%)`,
  values: [["A", "100"], ["B", "60"]],
  changed_in: ["B"],
  candidates: [
    { cause: "stock_split", fit: "ruled_out", why: "a stock split changes share counts", evidence: [] },
    { cause: "discontinued_operations", fit: "possible", why: "no numeric check", evidence: ["spin-off"] },
    { cause: "accounting_change", fit: "no_evidence", why: "not mentioned", evidence: [] },
    { cause: "reclassification", fit: "no_evidence", why: "not mentioned", evidence: [] },
  ],
  research_rounds_left: 2,
});

const waiting: Analysis = {
  thread_id: "t1", status: "awaiting_review", tickers: ["GE"], branches: 12, branches_done: 12, facts: 52, rejected: 0,
  errors: {}, cost_usd: 0.11, paid_usd: 0, cached_branches: 12, discrepancies: 13,
  resolutions: {
    "GE:revenue:FY2021": { cause: "discontinued_operations", verified: true, needs_review: true, lane: "review" },
    "GE:cash:FY2022": { cause: "discontinued_operations", verified: true, needs_review: true, lane: "review" },
    "GE:net_income:FY2023": { cause: "immaterial_revision", verified: true, needs_review: false, lane: null },
  },
  awaiting_review: [card("GE:revenue:FY2021"), card("GE:cash:FY2022")],
  decisions: {}, research_rounds: 0, memo: null, publication: null, last_run_id: null, runs: [],
  review_token: "tok-1", review_problems: {},
};

class FakeEventSource {
  onmessage: unknown = null;
  onerror: unknown = null;
  addEventListener() {}
  close() {}
}

describe("AnalysisView", () => {
  let posted: unknown[] = [];
  beforeEach(() => {
    posted = [];
    vi.stubGlobal("EventSource", FakeEventSource);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if (init?.method === "POST") {
          posted.push(JSON.parse(String(init.body)));
          return new Response(JSON.stringify({ status: "resuming" }), { status: 202 });
        }
        return new Response(JSON.stringify(url.endsWith("/t1") ? waiting : []));
      }),
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it("progress shows the lane, never the agent's cause", () => {
    const s0 = START;
    const s1 = step(s0, { event: "reconciled", id: "GE:net_income:FY2022", cause: "discontinued_operations", lane: "review" });
    expect(s1.toReview).toBe(1);
    expect(s1.last).not.toMatch(/discontinued/);
  });

  it("submits only when every card is decided, with exactly one decision per card", async () => {
    render(<AnalysisView threadId="t1" />);
    const submit = await screen.findByRole("button", { name: "Submit decisions" });
    expect(screen.getByText("1 decided by code (no human needed)")).toBeInTheDocument();
    expect(submit).toBeDisabled();

    const [first, second] = screen.getAllByRole("article");
    await userEvent.click(within(first!).getByRole("radio", { name: "discontinued operations" }));
    expect(submit).toBeDisabled(); // one of two
    await userEvent.click(within(second!).getByRole("radio", { name: "Escalate" }));
    expect(submit).toBeEnabled();

    await userEvent.click(submit);
    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0]).toEqual({
      decisions: {
        "GE:revenue:FY2021": { action: "accept", correct_cause: "discontinued_operations" },
        "GE:cash:FY2022": { action: "escalate" },
      },
      review_token: "tok-1", // a review made on cards that changed since is refused (409)
    });
  });
});

describe("humanError", () => {
  it("shows a stale-review refusal as its sentence", () => {
    expect(humanError(new Error('409: {"detail": "the cards changed; reload"}'))).toBe(
      "the cards changed; reload",
    );
  });
  it("lists invalid and missing decisions by card", () => {
    const body = JSON.stringify({
      detail: { invalid_decisions: { "GE:a:FY1": "accept needs correct_cause" }, missing_decisions: ["GE:b:FY2"] },
    });
    expect(humanError(new Error(`422: ${body}`))).toBe(
      "GE:a:FY1: accept needs correct_cause; no decision for GE:b:FY2",
    );
  });
  it("passes anything else through", () => {
    expect(humanError(new Error("network down"))).toBe("network down");
    expect(humanError(new Error("500: not json"))).toBe("500: not json");
  });
});


describe("AnalysisView on real API responses", () => {
  // Generated from the real graph (tests/support/api_fixtures.py); stale fixtures fail a Python test.
  const serve = (body: unknown) =>
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body))));
  afterEach(() => vi.unstubAllGlobals());

  it("a card back from research shows the findings and the round", async () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    serve(researched);
    render(<AnalysisView threadId="fixture-thread" />);
    expect(await screen.findByText("1 change needs your call")).toBeInTheDocument();
    expect(screen.getByText(/other: error correction/)).toBeInTheDocument();
    expect(screen.getByText(/Research round 1 \(hint: check the 8-K\)/)).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Research further \(1 left\)/ })).toBeEnabled();
  });

  it("links the run to its LangSmith trace when tracing is on", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        new Response(JSON.stringify(url === "/meta" ? { langsmith_project_url: "https://smith.example/p/1" } : published)),
      ),
    );
    render(<AnalysisView threadId="fixture-thread" />);
    const link = await screen.findByRole("link", { name: /Open trace in LangSmith/ });
    expect(link).toHaveAttribute("href", `https://smith.example/p/1/r/${published.last_run_id}`);
  });

  it("a published run shows the audit record, the decision and the memo", async () => {
    serve(published);
    render(<AnalysisView threadId="fixture-thread" />);
    expect(await screen.findByText(/published, exactly\s+once/)).toBeInTheDocument();
    expect(screen.getByText("spin-off recast")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "AAA summary" })).toBeInTheDocument();
  });
});
