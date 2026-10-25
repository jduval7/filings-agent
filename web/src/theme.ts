export type Theme = "dark" | "light";
const KEY = "filings-review-theme";

/** Saved choice, else dark (the default look; the toggle remembers a light choice). */
export function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    /* storage unavailable (private mode): fall through */
  }
  return "dark";
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* not persisted: fine */
  }
}
