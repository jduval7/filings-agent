import { describe, expect, it } from "vitest";
import { formatAmount, titleOf } from "./format";

describe("format", () => {
  it("titles a change for a reader", () => {
    expect(titleOf("GE:net_income:FY2022")).toBe("GE · Net income · FY2022");
  });

  it("formats amounts the way analysts read them", () => {
    expect(formatAmount("225000000", "net_income")).toBe("$225.0M");
    expect(formatAmount("76555000000", "revenue")).toBe("$76.56B");
    expect(formatAmount("-6520000000", "net_income")).toBe("-$6.52B");
    expect(formatAmount("25070000000", "diluted_shares")).toBe("25.07B shares");
    expect(formatAmount("950", "revenue")).toBe("$950");
  });
});
