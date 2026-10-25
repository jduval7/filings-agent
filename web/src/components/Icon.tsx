// Minimal inline icons (no dependency). Status always pairs an icon shape with a text label, so it
// never relies on color alone (IBM Carbon's status pattern; Bloomberg's color-vision guidance).

const PATHS = {
  check: "M3.5 8.5l3 3 6-7",
  cross: "M4.5 4.5l7 7m0-7l-7 7",
  half: "M8 2.5a5.5 5.5 0 1 0 0 11z",
  dash: "M4 8h8",
  shield: "M8 1.8l5 2v3.6c0 3.2-2.1 5.6-5 6.8-2.9-1.2-5-3.6-5-6.8V3.8z",
  doc: "M4 1.8h5l3 3v9.4H4z M9 1.8v3h3",
  sun: "M8 4.8a3.2 3.2 0 1 1 0 6.4 3.2 3.2 0 0 1 0-6.4zM8 .8v1.6M8 13.6v1.6M.8 8h1.6M13.6 8h1.6M2.9 2.9l1.1 1.1M12 12l1.1 1.1M2.9 13.1L4 12M12 4l1.1-1.1",
  moon: "M12.8 9.6A5.6 5.6 0 0 1 6.4 3.2a5.6 5.6 0 1 0 6.4 6.4z",
  search: "M7 2.5a4.5 4.5 0 1 1 0 9 4.5 4.5 0 0 1 0-9zM10.4 10.4l3.1 3.1",
  flag: "M3.5 14.5V2M3.5 2.5h8l-1.8 3 1.8 3h-8",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 14 }: { name: IconName; size?: number }) {
  const filled = name === "half" || name === "moon";
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      aria-hidden="true"
      fill={filled ? "currentColor" : "none"}
      stroke={filled ? "none" : "currentColor"}
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      className="icon"
    >
      {name === "half" && (
        <circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" strokeWidth={1.4} />
      )}
      <path d={PATHS[name]} />
    </svg>
  );
}
