export interface ApiErrorDetail {
  code: string;
  message: string;
}

export interface ApiErrorResponse {
  detail: ApiErrorDetail;
}

export interface AttributeItem {
  attr_id: string;
  attr_type: string;
  attr_value: string;
}

export interface ArticleItem {
  article_id: string;
  prod_name: string | null;
  product_group_name: string | null;
  product_type_name: string | null;
  garment_group_name: string | null;
  colour_group_name: string | null;
  graphical_appearance_name: string | null;
  department_name: string | null;
  section_name: string | null;
  index_name: string | null;
  index_group_name: string | null;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation_type: string;
}

export interface TrendAttribute {
  source_week: number;
  target_week: number;
  attr_id: string;
  attr_type: string;
  attr_value: string;
  rank: number;
  heat_t: number;
  pred_share_t1: number | null;
  pred_target_growth: number | null;
  is_trend_eligible_t: boolean;
}

export interface TrendListResponse {
  source_week: number | null;
  target_week: number | null;
  items: TrendAttribute[];
}

export interface HeatSeriesPoint {
  attr_id: string;
  attr_type: string;
  attr_value: string;
  week_id: number;
  heat: number;
  actual_target_growth: number | null;
  pred_target_growth: number | null;
  pred_share_t1: number | null;
}

export interface AttributeDetailResponse {
  attr_id: string;
  attr_type: string;
  attr_value: string;
  latest_trend: TrendAttribute | null;
  latest_heat: HeatSeriesPoint | null;
}

export interface HeatSeriesResponse {
  attr_id: string;
  points: HeatSeriesPoint[];
}

export interface AttributeArticlesResponse {
  attr_id: string;
  items: ArticleItem[];
}

export interface AttributeGraphResponse {
  attr_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ArticleSearchResponse {
  items: ArticleItem[];
}

export interface ArticleDetailResponse {
  article: ArticleItem;
  attributes: AttributeItem[];
}

export interface ArticleGraphResponse {
  article: GraphNode;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface DemoUserItem {
  case_id: string;
  customer_id: string;
  split: string;
  cutoff_week: number;
  label_week: number;
  hit_count: number;
  primary_tags: string;
  profile_summary: string;
  recommendation_summary: string;
}

export interface DemoUserListResponse {
  items: DemoUserItem[];
}

export interface UserProfileAttribute {
  case_id: string;
  customer_id: string;
  attr_id: string;
  attr_type: string;
  attr_value: string;
  preference_score: number;
  purchase_count: number;
  last_purchase_week: number;
}

export interface UserProfileResponse {
  case_id: string;
  items: UserProfileAttribute[];
}

export interface RecommendationItem {
  case_id: string;
  customer_id: string;
  article_id: string;
  rank: number;
  score: number;
  is_hit: boolean;
  candidate_sources: string;
  article: ArticleItem;
}

export interface RecommendationListResponse {
  case_id: string;
  items: RecommendationItem[];
}

export interface ScoreComponents {
  pop_score: number;
  sim_score: number;
  trend_score: number;
  recent_score: number;
  final_score: number;
}

export interface RecommendationExplanationResponse {
  case_id: string;
  article: ArticleItem;
  user_profile: UserProfileAttribute[];
  item_attributes: AttributeItem[];
  score_components: ScoreComponents;
  matching_trend_attributes: TrendAttribute[];
}

export interface MetricItem {
  metric_domain: string;
  model_or_method: string;
  split: string;
  metric_name: string;
  metric_value: number;
  display_order: number;
}

export interface MetricsListResponse {
  items: MetricItem[];
}

export interface MetricsSummaryResponse {
  groups: Record<string, MetricItem[]>;
}
