import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import type { Draft } from "../decisions";
import type { Card } from "../types";
import { ReviewCard } from "./ReviewCard";

const GE: Card = {
  id: "GE:net_income:FY2022",
  change: "net_income FY2022: 225,000,000 -> 336,000,000 (+49.3%)",
  values: [["0000040545-23-000023", "225000000"], ["0000040545-24-000027", "339000000"]],
  changed_in: ["0000040545-24-000027"],
  candidates: [
    { cause: "stock_split", fit: "ruled_out", why: "a stock split changes share counts, not dollar amounts", evidence: [] },
    {
      cause: "discontinued_operations", fit: "ruled_out",
      why: "net income attributable to the company includes discontinued operations",
      evidence: ["On January 3, 2023, we completed the separation of the HealthCare business"],
    },
    {
      cause: "accounting_change", fit: "possible", why: "accounting_change can move net_income; no numeric check",
      evidence: ["On January 1, 2023, we adopted ASU No. 2018-12"],
      research: [{
        stance: "supports", quote: "we adopted ASU No. 2018-12", source: "https://www.sec.gov/x",
        read: "2026-10-22", unverified_reading: "LDTI restated 2022",
      }],
    },
    { cause: "reclassification", fit: "possible", why: "no numeric check", evidence: [] },
  ],
  research_rounds_left: 0,
};

function Harness({ card }: { card: Card }) {
  const [draft, setDraft] = useState<Draft | undefined>();
  return <ReviewCard card={card} draft={draft} onChange={setDraft} />;
}

describe("ReviewCard", () => {
  it("shows every cause with code's verdict and the reason", () => {
    render(<Harness card={GE} />);
    expect(screen.getAllByText("ruled out")).toHaveLength(2);
    expect(screen.getByText(/includes discontinued operations/)).toBeInTheDocument();
    expect(screen.getByText(/we completed the separation/)).toBeInTheDocument();
  });

  it("keeps the investigator's reading collapsed and labelled unverified", () => {
    render(<Harness card={GE} />);
    const summary = screen.getByText("investigator's reading (unverified)");
    expect(summary.closest("details")).not.toHaveAttribute("open");
  });

  it("warns when the pick overrides a code rule", async () => {
    render(<Harness card={GE} />);
    await userEvent.click(screen.getByRole("radio", { name: "discontinued operations" }));
    expect(screen.getByText(/Code rules this out/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("radio", { name: "accounting change" }));
    expect(screen.queryByText(/Code rules this out/)).not.toBeInTheDocument();
  });

  it("offers research only while rounds are left", () => {
    render(<Harness card={GE} />);
    expect(screen.getByRole("radio", { name: /Research further \(0 left\)/ })).toBeDisabled();
  });

  it("summarizes research rounds, with the investigators' notes collapsed", () => {
    const researched: Card = {
      ...GE,
      research_rounds: [{
        round: 1, hint: "check the 10-K/A", looked_at: "Read both changed filings",
        open_questions: "quantify ASU vs spin-off", findings: 21, discarded_unverified: 2, error: null,
      }],
    };
    render(<Harness card={researched} />);
    expect(screen.getByText(/21 verified findings, 2 discarded as unverifiable/)).toBeInTheDocument();
    expect(screen.getByText(/hint: check the 10-K\/A/)).toBeInTheDocument();
    const notes = screen.getByText("where the investigators looked, and what's still open");
    expect(notes.closest("details")).not.toHaveAttribute("open");
  });
});
