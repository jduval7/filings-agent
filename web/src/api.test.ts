import { afterEach, describe, expect, it, vi } from "vitest";
import { listAnalyses, startAnalysis } from "./api";

function reply(status: number, body: unknown = []): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

afterEach(() => vi.unstubAllGlobals());

const sent = (mock: ReturnType<typeof vi.fn<typeof fetch>>, i: number): unknown =>
  JSON.parse(String(mock.mock.calls[i]?.[1]?.body));

describe("signing in", () => {
  it("asks for the token once on 401, trades it for a session, and retries", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(reply(401))
      .mockResolvedValueOnce(reply(200, { user: "alice" }))
      .mockResolvedValueOnce(reply(200, []));
    vi.stubGlobal("fetch", fetchMock);
    const prompt = vi.spyOn(window, "prompt").mockReturnValue("alice-token-0123456789");
    await expect(listAnalyses()).resolves.toEqual([]);
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/auth/session");
    expect(sent(fetchMock, 1)).toEqual({ token: "alice-token-0123456789" });
  });

  it("gives up without a token instead of looping", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockImplementation(() => Promise.resolve(reply(401, { detail: "no" }))));
    vi.spyOn(window, "prompt").mockReturnValue(null);
    await expect(listAnalyses()).rejects.toThrow("401");
  });
});

describe("re-analyze", () => {
  it("sends fresh only when asked", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(() => Promise.resolve(reply(202, { thread_id: "t" })));
    vi.stubGlobal("fetch", fetchMock);
    await startAnalysis(["GE"]);
    await startAnalysis(["GE"], true);
    expect(sent(fetchMock, 0)).toEqual({ tickers: ["GE"] });
    expect(sent(fetchMock, 1)).toEqual({ tickers: ["GE"], fresh: true });
  });
});
