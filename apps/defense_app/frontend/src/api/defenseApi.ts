import { apiGet } from "@/api/client";
import type {
  ArticleDetailResponse,
  ArticleGraphResponse,
  ArticleSearchResponse,
  AttributeArticlesResponse,
  AttributeDetailResponse,
  AttributeGraphResponse,
  DemoUserItem,
  DemoUserListResponse,
  HeatSeriesResponse,
  MetricsListResponse,
  MetricsSummaryResponse,
  RecommendationExplanationResponse,
  RecommendationListResponse,
  TrendEvidenceResponse,
  TrendListResponse,
  TrendSourceWeeksResponse,
  TrendSummaryResponse,
  UserProfileResponse,
} from "@/types/api";

export function listTrends(params?: {
  source_week?: number;
  attr_type?: string;
  limit?: number;
}) {
  return apiGet<TrendListResponse>("/api/trends", { params });
}

export function listTrendSourceWeeks() {
  return apiGet<TrendSourceWeeksResponse>("/api/trends/source-weeks");
}

export function getTrendSummary(params?: {
  source_week?: number;
  limit?: number;
}) {
  return apiGet<TrendSummaryResponse>("/api/trends/summary", { params });
}

export function getTrendEvidence(params?: {
  source_week?: number;
  limit?: number;
}) {
  return apiGet<TrendEvidenceResponse>("/api/trends/evidence", { params });
}

export function getTrendDetail(params?: {
  source_week?: number;
  attr_type?: string;
  limit?: number;
}) {
  return apiGet<TrendListResponse>("/api/trends/detail", { params });
}

export function getAttribute(attrId: string, sourceWeek?: number) {
  return apiGet<AttributeDetailResponse>(
    `/api/attributes/${encodeAttributePathParam(attrId)}`,
    {
      params: { source_week: sourceWeek },
    },
  );
}

export function getAttributeHeatSeries(
  attrId: string,
  params?: { source_week?: number; weeks?: number },
) {
  return apiGet<HeatSeriesResponse>(
    `/api/attributes/${encodeAttributePathParam(attrId)}/heat-series`,
    { params },
  );
}

export function getAttributeArticles(attrId: string, limit?: number) {
  return apiGet<AttributeArticlesResponse>(
    `/api/attributes/${encodeAttributePathParam(attrId)}/articles`,
    { params: { limit } },
  );
}

export function getAttributeGraph(attrId: string) {
  return apiGet<AttributeGraphResponse>(
    `/api/attributes/${encodeAttributePathParam(attrId)}/graph`,
  );
}

export function searchArticles(q: string, limit?: number) {
  return apiGet<ArticleSearchResponse>("/api/articles/search", {
    params: { q, limit },
  });
}

export function getArticle(articleId: string) {
  return apiGet<ArticleDetailResponse>(
    `/api/articles/${encodeURIComponent(articleId)}`,
  );
}

export function getArticleGraph(articleId: string) {
  return apiGet<ArticleGraphResponse>(
    `/api/articles/${encodeURIComponent(articleId)}/graph`,
  );
}

export function listDemoUsers(params?: {
  q?: string;
  tag?: string;
  limit?: number;
}) {
  return apiGet<DemoUserListResponse>("/api/demo-users", { params });
}

export function getDemoUser(caseId: string) {
  return apiGet<DemoUserItem>(`/api/demo-users/${encodeURIComponent(caseId)}`);
}

export function getUserProfile(caseId: string) {
  return apiGet<UserProfileResponse>(
    `/api/demo-users/${encodeURIComponent(caseId)}/profile`,
  );
}

export function getRecommendations(caseId: string) {
  return apiGet<RecommendationListResponse>(
    `/api/demo-users/${encodeURIComponent(caseId)}/recommendations`,
  );
}

export function getRecommendationExplanation(
  caseId: string,
  articleId: string,
) {
  return apiGet<RecommendationExplanationResponse>(
    `/api/demo-users/${encodeURIComponent(
      caseId,
    )}/recommendations/${encodeURIComponent(articleId)}/explanation`,
  );
}

export function getMetricsSummary() {
  return apiGet<MetricsSummaryResponse>("/api/metrics/summary");
}

export function listTrendMetrics(split?: string) {
  return apiGet<MetricsListResponse>("/api/metrics/trend", {
    params: { split },
  });
}

export function listRecommendationMetrics(split?: string) {
  return apiGet<MetricsListResponse>("/api/metrics/recommendation", {
    params: { split },
  });
}

function encodeAttributePathParam(attrId: string) {
  return encodeURIComponent(encodeURIComponent(attrId));
}
