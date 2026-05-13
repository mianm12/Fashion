<template>
  <section class="route-page">
    <header class="page-header">
      <p class="eyebrow">Attribute</p>
      <h1>{{ detail?.attr_value ?? attrId }}</h1>
      <div class="header-metrics">
        <span>{{ detail?.attr_type ?? "attr_type" }}</span>
        <span>Rank {{ detail?.latest_trend?.rank ?? "--" }}</span>
        <span>Week {{ detail?.latest_trend?.source_week ?? "--" }}</span>
      </div>
    </header>

    <div v-if="loading" class="dense-state">加载属性详情...</div>
    <div v-else-if="error" class="dense-state error-state">{{ error }}</div>

    <template v-else>
      <div v-if="auxiliaryError" class="dense-state error-state compact-state">
        {{ auxiliaryError }}
      </div>

      <div class="skeleton-grid two-columns">
        <article class="panel">
          <header class="panel-heading">
            <div>
              <h2>属性概览</h2>
              <p>{{ attrId }}</p>
            </div>
            <span class="count-pill">{{ detail?.attr_type ?? "--" }}</span>
          </header>
          <dl class="detail-grid">
            <div>
              <dt>Heat</dt>
              <dd>{{ formatNumber(detail?.latest_heat?.heat) }}</dd>
            </div>
            <div>
              <dt>Pred growth</dt>
              <dd>{{ formatPercent(detail?.latest_trend?.pred_target_growth) }}</dd>
            </div>
            <div>
              <dt>Pred share</dt>
              <dd>{{ formatPercent(detail?.latest_trend?.pred_share_t1) }}</dd>
            </div>
            <div>
              <dt>Eligible</dt>
              <dd>{{ detail?.latest_trend?.is_trend_eligible_t ? "yes" : "no" }}</dd>
            </div>
          </dl>
        </article>

        <article class="panel">
          <header class="panel-heading">
            <div>
              <h2>属性关系</h2>
              <p>parent / child edges</p>
            </div>
            <span class="count-pill">{{ graph?.edges.length ?? 0 }} edges</span>
          </header>
          <div v-if="!graph || graph.edges.length === 0" class="dense-state compact-state">
            暂无父子属性关系
          </div>
          <ul v-else class="relation-list">
            <li v-for="edge in graph.edges" :key="`${edge.source}-${edge.target}`">
              <span>{{ edge.source }}</span>
              <strong>{{ edge.relation_type }}</strong>
              <span>{{ edge.target }}</span>
            </li>
          </ul>
        </article>
      </div>

      <article class="panel chart-panel">
        <header class="panel-heading">
          <div>
            <h2>最近 8 周热度与增长</h2>
            <p>heat / actual growth / predicted growth</p>
          </div>
          <span class="count-pill">{{ heatSeries.length }} points</span>
        </header>
        <AttributeHeatChart :points="heatSeries" />
      </article>

      <article class="panel">
        <header class="panel-heading">
          <div>
            <h2>关联商品样例</h2>
            <p>articles carrying this attribute</p>
          </div>
          <span class="count-pill">{{ articles.length }} items</span>
        </header>
        <div v-if="articles.length === 0" class="dense-state compact-state">
          暂无关联商品
        </div>
        <div v-else class="article-table">
          <RouterLink
            v-for="article in articles"
            :key="article.article_id"
            class="article-row"
            :to="{ name: 'article-graph', params: { articleId: article.article_id } }"
          >
            <span>{{ article.article_id }}</span>
            <strong>{{ article.prod_name ?? article.product_type_name ?? "--" }}</strong>
            <span>{{ article.colour_group_name ?? "--" }}</span>
            <span>{{ article.product_type_name ?? "--" }}</span>
          </RouterLink>
        </div>
      </article>
    </template>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";

import {
  getAttribute,
  getAttributeArticles,
  getAttributeGraph,
  getAttributeHeatSeries,
} from "@/api/defenseApi";
import AttributeHeatChart from "@/components/attribute/AttributeHeatChart.vue";
import type {
  ArticleItem,
  AttributeDetailResponse,
  AttributeGraphResponse,
  HeatSeriesPoint,
} from "@/types/api";

const props = defineProps<{
  attrId: string;
}>();

const route = useRoute();
const detail = ref<AttributeDetailResponse | null>(null);
const heatSeries = ref<HeatSeriesPoint[]>([]);
const articles = ref<ArticleItem[]>([]);
const graph = ref<AttributeGraphResponse | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);
const auxiliaryError = ref<string | null>(null);
let attributeRequestId = 0;

onMounted(() => {
  void loadAttribute();
});

watch(
  () => [props.attrId, route.query.source_week],
  () => {
    void loadAttribute();
  },
);

async function loadAttribute() {
  const requestId = ++attributeRequestId;
  loading.value = true;
  error.value = null;
  auxiliaryError.value = null;
  detail.value = null;
  heatSeries.value = [];
  articles.value = [];
  graph.value = null;
  const sourceWeek = readOptionalInt(route.query.source_week);

  try {
    const detailResponse = await getAttribute(props.attrId, sourceWeek ?? undefined);
    if (requestId !== attributeRequestId) {
      return;
    }
    detail.value = detailResponse;

    const [heatResult, articleResult, graphResult] = await Promise.allSettled([
      getAttributeHeatSeries(props.attrId, {
        source_week: sourceWeek ?? undefined,
        weeks: 8,
      }),
      getAttributeArticles(props.attrId, 12),
      getAttributeGraph(props.attrId),
    ]);

    if (requestId !== attributeRequestId) {
      return;
    }

    if (heatResult.status === "fulfilled") {
      heatSeries.value = heatResult.value.points;
    }
    if (articleResult.status === "fulfilled") {
      articles.value = articleResult.value.items;
    }
    if (graphResult.status === "fulfilled") {
      graph.value = graphResult.value;
    }
    const failedAuxiliaryLoads = [
      heatResult,
      articleResult,
      graphResult,
    ].filter((result) => result.status === "rejected").length;
    if (failedAuxiliaryLoads > 0) {
      auxiliaryError.value = "部分辅助数据加载失败，已保留可用属性信息";
    }
  } catch (loadError) {
    if (requestId !== attributeRequestId) {
      return;
    }
    error.value = getErrorMessage(loadError);
  } finally {
    if (requestId === attributeRequestId) {
      loading.value = false;
    }
  }
}

function readOptionalInt(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function formatNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(3)
    : "--";
}

function formatPercent(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%`
    : "--";
}

function getErrorMessage(value: unknown) {
  return value instanceof Error ? value.message : "请求失败";
}
</script>
