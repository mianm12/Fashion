export function formatNumber(value: number | null | undefined, digits = 3) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "--";
}

export function formatInteger(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.round(value).toLocaleString("en-US")
    : "--";
}

export function formatPercent(value: number | null | undefined, digits = 1) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : "--";
}

export function formatSignedPercent(
  value: number | null | undefined,
  digits = 1,
) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  const formatted = formatPercent(value, digits);
  return value > 0 ? `+${formatted}` : formatted;
}

export function formatScore(value: number | null | undefined, digits = 4) {
  return formatNumber(value, digits);
}

export function formatWeek(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `W${value}` : "--";
}

export function parseJsonStringArray(value: string) {
  try {
    const parsed = JSON.parse(value) as unknown;
    if (Array.isArray(parsed)) {
      return parsed.filter((item): item is string => typeof item === "string");
    }
  } catch {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return [];
}
