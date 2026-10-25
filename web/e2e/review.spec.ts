import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

/** A fresh GE run (a new Idempotency-Key each time), opened in the review UI. */
async function freshGeRun(page: Page): Promise<string> {
  const resp = await page.request.post("/analyses", {
    data: { tickers: ["GE"] },
    headers: { "Idempotency-Key": `e2e-${Date.now()}` },
  });
  expect(resp.ok()).toBeTruthy();
  const { thread_id } = (await resp.json()) as { thread_id: string };
  await page.goto(`/#/${thread_id}`);
  return thread_id;
}

test("a reviewer decides a GE run end to end: escalate, research, pick, publish", async ({ page }) => {
  await freshGeRun(page);
  await expect(page.getByRole("heading", { name: "10 changes need your call" })).toBeVisible();
  const cards = page.getByRole("article");
  await expect(cards).toHaveCount(10);

  // The card never shows the agent's answer; code's verdicts and citations are visible.
  const ni = page.getByRole("article", { name: "GE:net_income:FY2022" });
  await expect(ni.getByText(/ASC 205-20-45-3A/)).toBeVisible();

  const submit = page.getByRole("button", { name: "Submit decisions" });
  for (const card of await cards.all()) {
    if ((await card.getAttribute("aria-label")) === "GE:net_income:FY2022") {
      await card.getByText(/Research further/).click();
    } else {
      await card.getByText("Escalate", { exact: true }).click();
    }
  }
  await expect(page.getByText("10 of 10 decided")).toBeVisible();
  await expect(submit).toBeEnabled();
  await submit.click();

  // The researched card comes back alone, with findings and a round left (cached: seconds).
  await expect(page.getByRole("heading", { name: "1 change needs your call" })).toBeVisible();
  await expect(page.getByText(/Research round 1/)).toBeVisible();
  const back = page.getByRole("article", { name: "GE:net_income:FY2022" });
  await back.getByRole("radio", { name: "discontinued operations" }).click();
  await expect(back.getByText(/Code rules this out/)).toBeVisible(); // a warning, not a block
  await back.getByRole("radio", { name: "accounting change" }).click();
  await expect(back.getByText(/Code rules this out/)).toHaveCount(0);
  await page.getByRole("button", { name: "Submit decisions" }).click();

  await expect(page.getByRole("heading", { name: "Published" })).toBeVisible();
  await expect(page.getByText(/published, exactly\s+once/)).toBeVisible();
  const rows = page.locator("table.decisions tbody tr");
  await expect(rows).toHaveCount(10);
  await expect(page.locator("tr.escalated")).toHaveCount(9);
});

test("the theme toggle switches and is remembered", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark"); // dark by default
  await page.getByRole("button", { name: "switch to light mode" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
});

for (const theme of ["dark", "light"] as const) {
  test(`no serious accessibility violations on a review page (${theme})`, async ({ page }) => {
    await page.addInitScript((t) => localStorage.setItem("filings-review-theme", t), theme);
    await freshGeRun(page);
    await expect(page.getByRole("heading", { name: "10 changes need your call" })).toBeVisible();
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    const detail = serious.flatMap((v) => v.nodes.map((n) => `${v.id}: ${n.target.join(" ")} ${n.failureSummary ?? ""}`));
    expect(detail).toEqual([]);
  });
}
