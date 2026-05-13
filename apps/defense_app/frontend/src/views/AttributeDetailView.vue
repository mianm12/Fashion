<template>
  <section class="route-page attribute-detail-page">
    <PageToolbar :title="'属性详情'" :context="toolbarContext" :status="toolbarStatus">
      <template #actions>
        <RouterLink class="toolbar-button" :to="trendRoute">
          <ArrowLeft aria-hidden="true" />
          返回趋势看板
        </RouterLink>
        <label class="compact-field">
          <span>数据源周</span>
          <select v-model.number="sourceWeekInput">
            <option
              v-for="week in resolvedSourceWeeks"
              :key="week"
              :value="week"
            >
              第 {{ week }} 周
            </option>
          </select>
        </label>
        <button type="button" class="toolbar-button" @click="loadAttribute">
          <RefreshCw aria-hidden="true" />
          刷新
        </button>
      </template>
    </PageToolbar>

    <StatusBlock
      v-if="sourceWeekError"
      type="error"
      title="数据源周加载失败"
      :message="sourceWeekError"
      class="compact-state"
    />
    <StatusBlock
      v-if="loading && !detail"
      type="loading"
      title="加载属性详情..."
      class="compact-state"
    />
    <StatusBlock
      v-else-if="error"
      type="error"
      title="属性详情加载失败"
      :message="error"
      class="compact-state"
    />

    <template v-else>
      <StatusBlock
        v-if="auxiliaryError"
        type="error"
        title="部分辅助数据加载失败"
        :message="auxiliaryError"
        class="compact-state"
      />

      <AttributeSummaryBand :detail="detail" />

      <Panel
        title="最近 8 周热度与增长"
        :subtitle="`热度、预测增长与真实增长 · ${currentWindowLabel}`"
        class="attribute-heat-panel"
      >
        <template #actions>
          <span class="count-pill">{{ heatSeries.length }} 个点</span>
        </template>
        <AttributeHeatChart :points="heatSeries" :source-week="sourceWeekInput" />
      </Panel>

      <div class="attribute-related-grid">
        <Panel title="关联商品样例" subtitle="带有该属性的代表商品">
          <template #actions>
            <span class="count-pill">{{ articles.length }} 件商品</span>
          </template>
          <StatusBlock
            v-if="articles.length === 0"
            title="暂无关联商品"
            class="compact-state"
          />
            <div v-else class="article-evidence-table">
              <div class="article-evidence-row article-evidence-head">
                <span>商品 ID</span>
                <span>商品名称</span>
                <span>品类</span>
                <span>颜色</span>
                <span>服装组</span>
                <span>操作</span>
              </div>
              <div
                v-for="article in articles"
                :key="article.article_id"
                class="article-evidence-row"
              >
                <span>{{ article.article_id }}</span>
                <strong>{{ article.prod_name ?? "--" }}</strong>
                <span>{{ article.product_type_name ?? "--" }}</span>
                <span>{{ article.colour_group_name ?? "--" }}</span>
                <span>{{ article.garment_group_name ?? "--" }}</span>
                <RouterLink
                  :to="{
                    name: 'article-graph',
                    params: { articleId: article.article_id },
                    query: graphQuery,
                  }"
                >
                  属性图
                </RouterLink>
              </div>
          </div>
        </Panel>

        <Panel title="属性关系树" subtitle="父级、当前属性与子级关系">
          <template #actions>
            <RouterLink
              class="panel-action-link"
              :to="{ name: 'article-graph', query: graphQuery }"
            >
              查看完整属性图
            </RouterLink>
          </template>
          <AttributeRelationTree
            :attr-id="props.attrId"
            :graph="graph"
            :source-week="sourceWeekInput"
          />
        </Panel>
      </div>

      <Panel title="周级明细" subtitle="用于追问时审计预测方向和真实方向">
        <AttributeWeekDetailTable :points="heatSeries" :source-week="sourceWeekInput" />
      </Panel>
    </template>
  </section>
</template>

<script setup lang="ts">
import { ArrowLeft, RefreshCw } from "lucide-vue-next";
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";

import {
  getAttribute,
  getAttributeArticles,
  getAttributeGraph,
  getAttributeHeatSeries,
  listTrendSourceWeeks,
} from "@/api/defenseApi";
import AttributeHeatChart from "@/components/attribute/AttributeHeatChart.vue";
import AttributeRelationTree from "@/components/attribute/AttributeRelationTree.vue";
import AttributeSummaryBand from "@/components/attribute/AttributeSummaryBand.vue";
import AttributeWeekDetailTable from "@/components/attribute/AttributeWeekDetailTable.vue";
import PageToolbar from "@/components/layout/PageToolbar.vue";
import Panel from "@/components/ui/Panel.vue";
import StatusBlock from "@/components/ui/StatusBlock.vue";
import type {
  ArticleItem,
  AttributeDetailResponse,
  AttributeGraphResponse,
  HeatSeriesPoint,
} from "@/types/api";
import { formatWeek } from "@/utils/formatters";

const props = defineProps<{
  attrId: string;
}>();

const route = useRoute();
const router = useRouter();
const detail = ref<AttributeDetailResponse | null>(null);
const heatSeries = ref<HeatSeriesPoint[]>([]);
const articles = ref<ArticleItem[]>([]);
const graph = ref<AttributeGraphResponse | null>(null);
const sourceWeeks = ref<number[]>([]);
const sourceWeekInput = ref<number | null>(readOptionalInt(route.query.source_week));
const loading = ref(false);
const error = ref<string | null>(null);
const auxiliaryError = ref<string | null>(null);
const sourceWeekError = ref<string | null>(null);
let attributeRequestId = 0;
let isInitializing = true;

const resolvedSourceWeeks = computed(() => {
  const weeks = new Set(sourceWeeks.value);
  if (sourceWeekInput.value) {
    weeks.add(sourceWeekInput.value);
  }
  return [...weeks].sort((left, right) => left - right);
});

const toolbarContext = computed(() => {
  const value = detail.value?.attr_value ?? props.attrId;
  const type = detail.value?.attr_type ?? "attr_type";
  const targetWeek =
    detail.value?.latest_trend?.target_week ??
    (sourceWeekInput.value ? sourceWeekInput.value + 1 : null);
  return `${value} · ${type} · 预测第 ${targetWeek ?? "--"} 周`;
});

const toolbarStatus = computed(() =>
  detail.value?.latest_trend ? "趋势证据就绪" : "属性查询",
);

const currentWindowLabel = computed(() =>
  sourceWeekInput.value
    ? `${formatWeek(sourceWeekInput.value)} -> ${formatWeek(sourceWeekInput.value + 1)}`
    : "--",
);

const trendRoute = computed(() => ({
  name: "trends",
  query: sourceWeekInput.value ? { source_week: sourceWeekInput.value } : {},
}));

const graphQuery = computed(() => ({
  attr_id: props.attrId,
  ...(sourceWeekInput.value ? { source_week: sourceWeekInput.value } : {}),
}));

onMounted(async () => {
  await loadSourceWeeks();
  isInitializing = false;
  await updateQuery();
  await loadAttribute();
});

watch(
  () => props.attrId,
  () => {
    void loadAttribute();
  },
);

watch(sourceWeekInput, () => {
  if (isInitializing) {
    return;
  }
  updateQuery();
  void loadAttribute();
});

async function loadSourceWeeks() {
  sourceWeekError.value = null;
  try {
    const response = await listTrendSourceWeeks();
    sourceWeeks.value = response.items;
    if (sourceWeekInput.value === null) {
      sourceWeekInput.value = response.default_source_week ?? response.items.at(-1) ?? null;
    }
  } catch (loadError) {
    sourceWeekError.value = getErrorMessage(loadError);
  }
}

async function loadAttribute() {
  const requestId = ++attributeRequestId;
  loading.value = true;
  error.value = null;
  auxiliaryError.value = null;
  detail.value = null;
  heatSeries.value = [];
  articles.value = [];
  graph.value = null;

  try {
    const detailResponse = await getAttribute(
      props.attrId,
      sourceWeekInput.value ?? undefined,
    );
    if (requestId !== attributeRequestId) {
      return;
    }
    detail.value = detailResponse;

    const [heatResult, articleResult, graphResult] = await Promise.allSettled([
      getAttributeHeatSeries(props.attrId, {
        source_week: sourceWeekInput.value ?? undefined,
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
      auxiliaryError.value = "已保留可用属性信息，请刷新重试缺失区域";
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

function updateQuery() {
  const query: Record<string, string> = {};
  if (sourceWeekInput.value) {
    query.source_week = String(sourceWeekInput.value);
  }
  return router.replace({
    name: "attribute-detail",
    params: { attrId: props.attrId },
    query,
  });
}

function readOptionalInt(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function getErrorMessage(value: unknown) {
  return value instanceof Error ? value.message : "请求失败";
}
</script>
