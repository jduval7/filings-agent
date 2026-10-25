const METRIC_NAMES: Record<string, string> = {
  revenue: "Revenue",
  operating_income: "Operating income",
  net_income: "Net income",
  long_term_debt: "Long-term debt",
  cash: "Cash",
  diluted_shares: "Diluted shares",
  operating_cash_flow: "Operating cash flow",
};

/** "GE:net_income:FY2022" -> "GE · Net income · FY2022" */
export function titleOf(id: string): string {
  const [ticker, metric, fy] = id.split(":");
  return [ticker, METRIC_NAMES[metric ?? ""] ?? metric, fy].filter(Boolean).join(" · ");
}

/** Dollars as $225.0M / $1.35B; share counts as 2.51B shares. Negative dollars in parentheses-free
 *  minus form, as analysts read them in a memo. */
export function formatAmount(value: string | number, metric: string): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  const abs = Math.abs(n);
  const [scaled, unit] =
    abs >= 1e12 ? [n / 1e12, "T"] : abs >= 1e9 ? [n / 1e9, "B"] : abs >= 1e6 ? [n / 1e6, "M"] : [n, ""];
  const digits = unit ? (Math.abs(scaled) >= 100 ? 1 : 2) : 0;
  const text = `${Math.abs(scaled).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}${unit}`;
  if (metric === "diluted_shares") return `${n < 0 ? "-" : ""}${text} shares`;
  return `${n < 0 ? "-" : ""}$${text}`;
}
