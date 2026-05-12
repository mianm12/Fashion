import type { ApiErrorResponse } from "@/types/api";

export class DefenseApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "DefenseApiError";
    this.status = status;
    this.code = code;
  }
}

type QueryValue = string | number | boolean | null | undefined;

export interface RequestOptions {
  params?: Record<string, QueryValue>;
}

export async function apiGet<T>(
  path: `/api/${string}`,
  options: RequestOptions = {},
): Promise<T> {
  const response = await fetchJson(path, options);

  if (!response.ok) {
    throw await buildApiError(response);
  }

  try {
    return (await response.json()) as T;
  } catch (error) {
    throw new DefenseApiError(
      response.status,
      "invalid_response",
      "API 返回了无法解析的响应",
    );
  }
}

async function fetchJson(
  path: `/api/${string}`,
  options: RequestOptions,
): Promise<Response> {
  try {
    return await fetch(buildUrl(path, options.params), {
      headers: {
        Accept: "application/json",
      },
    });
  } catch (error) {
    throw new DefenseApiError(0, "network_error", "无法连接展示应用 API");
  }
}

function buildUrl(path: `/api/${string}`, params?: Record<string, QueryValue>) {
  const query = new URLSearchParams();

  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== null && value !== undefined && value !== "") {
      query.set(key, String(value));
    }
  }

  const serialized = query.toString();
  return serialized ? `${path}?${serialized}` : path;
}

async function buildApiError(response: Response) {
  const fallback = new DefenseApiError(
    response.status,
    "request_failed",
    `API request failed with status ${response.status}`,
  );

  try {
    const payload = (await response.json()) as Partial<ApiErrorResponse>;
    const detail = payload.detail;
    if (detail?.code && detail?.message) {
      return new DefenseApiError(response.status, detail.code, detail.message);
    }
  } catch {
    return fallback;
  }

  return fallback;
}
