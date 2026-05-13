<template>
  <section class="route-page trend-dashboard-page">
    <PageToolbar
      title="趋势看板"
      :context="`预测第 ${targetWeekLabel} 周上升趋势`"
      :status="summary?.model_status ?? 'LightGBM stable'"
    >
      <template #actions>
        <label class="compact-field">
          <span>数据源周</span>
          <select v-model.number="sourceWeekInput">
            <option
              v-for="week in sourceWeeks"
              :key="week"
              :value="week"
            >
              第 {{ week }} 周
            </option>
          </select>
        </label>
        <label class="compact-field">
          <span>Top-K</span>
          <select v-model.number="topK">
            <option :value="5">5</option>
            <option :value="10">10</option>
            <option :value="20">20</option>
            <option :value="50">50</option>
          </select>
        </label>
        <label class="compact-field attr-filter-field">
          <span>属性类型</span>
          <select v-model="selectedAttrType">
            <option
              v-for="item in filterItems"
              :key="item.value"
              :value="item.value"
            >
              {{ item.label }}
            </option>
          </select>
        </label>
        <button type="button" class="toolbar-button" @click="loadDashboard">
          <RefreshCw aria-hidden="true" />
          刷新
        </button>
      </template>
    </PageToolbar>

    <div v-if="sourceWeekError" class="dense-state error-state compact-state">
      {{ sourceWeekError }}
    </div>

    <TrendMetricStrip
      :summary="summary"
      :loading="summaryLoading"
      :error="summaryError"
    />

    <TrendRankMatrix
      :groups="trendGroups"
      :attr-type="selectedAttrType"
      :source-week="sourceWeekInput"
      :loading="trendLoading"
      :errors="trendErrors"
    />

    <TrendEvidenceArea
      :evidence="evidence"
      :loading="evidenceLoading"
      :error="evidenceError"
    />

    <TrendDetailTable
      :response="detail"
      :loading="detailLoading"
      :error="detailError"
    />
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { RefreshCw } from "lucide-vue-next";

import {
  getTrendDetail,
  getTrendEvidence,
  getTrendSummary,
  listTrendSourceWeeks,
  listTrends,
} from "@/api/defenseApi";
import PageToolbar from "@/components/layout/PageToolbar.vue";
import TrendDetailTable from "@/components/trend/TrendDetailTable.vue";
import TrendEvidenceArea from "@/components/trend/TrendEvidenceArea.vue";
import TrendMetricStrip from "@/components/trend/TrendMetricStrip.vue";
import TrendRankMatrix from "@/components/trend/TrendRankMatrix.vue";
import {
  attrTypeLabels,
  coreTrendAttrTypes,
  type CoreTrendAttrType,
  type TrendAttrFilter,
} from "@/components/trend/trendTypes";
import type {
  TrendAttribute,
  TrendEvidenceResponse,
  TrendListResponse,
  TrendSummaryResponse,
} from "@/types/api";

const route = useRoute();
const router = useRouter();

const sourceWeeks = ref<number[]>([]);
const sourceWeekInput = ref<number | null>(readOptionalInt(route.query.source_week));
const sourceWeekError = ref<string | null>(null);
const topK = ref(readPositiveInt(route.query.limit, 10));
const selectedAttrType = ref<TrendAttrFilter>(readAttrType(route.query.attr_type));

const summary = ref<TrendSummaryResponse | null>(null);
const evidence = ref<TrendEvidenceResponse | null>(null);
const detail = ref<TrendListResponse | null>(null);
const trendGroups = ref<Partial<Record<CoreTrendAttrType, TrendAttribute[]>>>({});
const trendErrors = ref<Partial<Record<CoreTrendAttrType, string | null>>>({});

const summaryLoading = ref(false);
const trendLoading = ref(false);
const evidenceLoading = ref(false);
const detailLoading = ref(false);
const summaryError = ref<string | null>(null);
const evidenceError = ref<string | null>(null);
const detailError = ref<string | null>(null);
let dashboardRequestId = 0;

const filterItems: Array<{ value: TrendAttrFilter; label: string }> = [
  { value: "all", label: "全部" },
  ...coreTrendAttrTypes.map((type) => ({
    value: type,
    label: attrTypeLabels[type],
  })),
];

const targetWeekLabel = computed(() =>
  summary.value?.target_week ??
  detail.value?.target_week ??
  (sourceWeekInput.value ? sourceWeekInput.value + 1 : "--"),
);

onMounted(async () => {
  await loadSourceWeeks();
  await loadDashboard();
});

watch([sourceWeekInput, topK, selectedAttrType], () => {
  updateQuery();
  void loadDashboard();
});

async function loadSourceWeeks() {
  sourceWeekError.value = null;
  try {
    const response = await listTrendSourceWeeks();
    sourceWeeks.value = response.items;
    if (sourceWeekInput.value === null) {
      sourceWeekInput.value = response.default_source_week ?? response.items.at(-1) ?? null;
    }
  } catch (error) {
    sourceWeekError.value = getErrorMessage(error);
  }
}

async function loadDashboard() {
  const requestId = ++dashboardRequestId;
  const params = {
    source_week: sourceWeekInput.value ?? undefined,
    limit: topK.value,
  };

  summaryLoading.value = true;
  trendLoading.value = true;
  evidenceLoading.value = true;
  detailLoading.value = true;
  summaryError.value = null;
  evidenceError.value = null;
  detailError.value = null;
  trendErrors.value = {};

  const [summaryResult, trendResult, evidenceResult, detailResult] =
    await Promise.allSettled([
      getTrendSummary(params),
      listTrends(params),
      getTrendEvidence(params),
      getTrendDetail({
        ...params,
        attr_type:
          selectedAttrType.value === "all" ? undefined : selectedAttrType.value,
      }),
    ]);

  if (requestId !== dashboardRequestId) {
    return;
  }

  if (summaryResult.status === "fulfilled") {
    summary.value = summaryResult.value;
  } else {
    summary.value = null;
    summaryError.value = getErrorMessage(summaryResult.reason);
  }

  if (trendResult.status === "fulfilled") {
    trendGroups.value = groupTrends(trendResult.value.items);
  } else {
    trendGroups.value = {};
    const message = getErrorMessage(trendResult.reason);
    trendErrors.value = Object.fromEntries(
      coreTrendAttrTypes.map((type) => [type, message]),
    );
  }

  if (evidenceResult.status === "fulfilled") {
    evidence.value = evidenceResult.value;
  } else {
    evidence.value = null;
    evidenceError.value = getErrorMessage(evidenceResult.reason);
  }

  if (detailResult.status === "fulfilled") {
    detail.value = detailResult.value;
  } else {
    detail.value = null;
    detailError.value = getErrorMessage(detailResult.reason);
  }

  summaryLoading.value = false;
  trendLoading.value = false;
  evidenceLoading.value = false;
  detailLoading.value = false;
}

function groupTrends(items: TrendAttribute[]) {
  const groups: Partial<Record<CoreTrendAttrType, TrendAttribute[]>> = {};
  for (const type of coreTrendAttrTypes) {
    groups[type] = items.filter((item) => item.attr_type === type);
  }
  return groups;
}

function updateQuery() {
  void router.replace({
    query: {
      ...route.query,
      source_week: sourceWeekInput.value ?? undefined,
      attr_type:
        selectedAttrType.value === "all" ? undefined : selectedAttrType.value,
      limit: topK.value,
    },
  });
}

function readOptionalInt(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function readPositiveInt(value: unknown, fallback: number) {
  const parsed = readOptionalInt(value);
  return parsed && parsed > 0 ? parsed : fallback;
}

function readAttrType(value: unknown): TrendAttrFilter {
  return typeof value === "string" &&
    coreTrendAttrTypes.includes(value as CoreTrendAttrType)
    ? (value as CoreTrendAttrType)
    : "all";
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "请求失败";
}
</script>
