import { beforeEach, describe, expect, it } from "vitest";
import { applyTheme, initialTheme } from "./theme";

describe("theme", () => {
  beforeEach(() => localStorage.clear());

  it("defaults to dark, and a saved choice wins", () => {
    expect(initialTheme()).toBe("dark");
    applyTheme("light");
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(initialTheme()).toBe("light"); // remembered across reloads
  });
});
