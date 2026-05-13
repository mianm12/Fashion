<template>
  <section class="route-page">
    <header class="page-header">
      <p class="eyebrow">Trend</p>
      <h1>趋势看板</h1>
      <div class="header-metrics">
        <label class="compact-field">
          <span>Source week</span>
          <input v-model.number="sourceWeekInput" type="number" min="1" />
        </label>
        <label class="compact-field">
          <span>Top K</span>
          <select v-model.number="topK">
            <option :value="5">5</option>
            <option :value="10">10</option>
            <option :value="20">20</option>
            <option :value="50">50</option>
          </select>
        </label>
      </div>
    </header>

    <TrendMetricStrip :groups="metrics" />

    <div class="filter-bar" aria-label="属性类型筛选">
      <button
        v-for="item in filterItems"
        :key="item.value"
        type="button"
        class="segment-button"
        :class="{ active: selectedAttrType === item.value }"
        @click="selectedAttrType = item.value"
      >
        {{ item.label }}
      </button>
    </div>

    <div v-if="metricsError" class="dense-state error-state">
      {{ metricsError }}
    </div>

    <div class="skeleton-grid four-columns trend-board-grid">
      <TrendBoard
        v-for="attrType in visibleAttrTypes"
        :key="attrType"
        :title="attrTypeLabels[attrType]"
        :attr-type="attrType"
        :items="trendGroups[attrType] ?? []"
        :loading="trendLoading"
        :error="trendErrors[attrType]"
      />
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";

import { getMetricsSummary, listTrends } from "@/api/defenseApi";
import TrendBoard from "@/components/trend/TrendBoard.vue";
import TrendMetricStrip from "@/components/trend/TrendMetricStrip.vue";
import type { MetricItem, TrendAttribute } from "@/types/api";

type AttrType =
  | "colour_group_name"
  | "product_type_name"
  | "graphical_appearance_name"
  | "garment_group_name";
type FilterValue = AttrType | "all";

const route = useRoute();
const router = useRouter();

const attrTypes: AttrType[] = [
  "colour_group_name",
  "product_type_name",
  "graphical_appearance_name",
  "garment_group_name",
];

const attrTypeLabels: Record<AttrType, string> = {
  colour_group_name: "颜色",
  product_type_name: "品类",
  graphical_appearance_name: "图案",
  garment_group_name: "服装组",
};

const filterItems: Array<{ value: FilterValue; label: string }> = [
  { value: "all", label: "全部" },
  ...attrTypes.map((attrType) => ({
    value: attrType,
    label: attrTypeLabels[attrType],
  })),
];

const metrics = ref<Record<string, MetricItem[]>>({});
const metricsError = ref<string | null>(null);
const trendGroups = ref<Partial<Record<AttrType, TrendAttribute[]>>>({});
const trendErrors = ref<Partial<Record<AttrType, string | null>>>({});
const trendLoading = ref(false);
const topK = ref(readPositiveInt(route.query.limit, 10));
const sourceWeekInput = ref<number | null>(readOptionalInt(route.query.source_week));
const selectedAttrType = ref(readAttrType(route.query.attr_type));
let trendRequestId = 0;

const visibleAttrTypes = computed(() =>
  selectedAttrType.value === "all"
    ? attrTypes
    : attrTypes.filter((attrType) => attrType === selectedAttrType.value),
);

onMounted(() => {
  void loadMetrics();
  void loadTrends();
});

watch([topK, sourceWeekInput], () => {
  updateQuery();
  void loadTrends();
});

watch(selectedAttrType, () => {
  updateQuery();
});

async function loadMetrics() {
  metricsError.value = null;
  try {
    metrics.value = (await getMetricsSummary()).groups;
  } catch (error) {
    metricsError.value = getErrorMessage(error);
  }
}

async function loadTrends() {
  const requestId = ++trendRequestId;
  trendLoading.value = true;
  trendErrors.value = {};

  const entries = await Promise.all(
    attrTypes.map(async (attrType) => {
      try {
        const response = await listTrends({
          source_week: sourceWeekInput.value ?? undefined,
          attr_type: attrType,
          limit: topK.value,
        });
        return [attrType, response.items, null] as const;
      } catch (error) {
        return [attrType, [], getErrorMessage(error)] as const;
      }
    }),
  );

  if (requestId !== trendRequestId) {
    return;
  }

  trendGroups.value = Object.fromEntries(
    entries.map(([attrType, items]) => [attrType, items]),
  );
  trendErrors.value = Object.fromEntries(
    entries.map(([attrType, , error]) => [attrType, error]),
  );
  trendLoading.value = false;
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

function readAttrType(value: unknown): AttrType | "all" {
  return typeof value === "string" && attrTypes.includes(value as AttrType)
    ? (value as AttrType)
    : "all";
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "请求失败";
}
</script>
