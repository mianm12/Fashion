<template>
  <section class="attribute-summary-band" aria-label="属性摘要指标">
    <MetricTile
      label="榜单排名"
      :value="rankValue"
      :hint="sourceWeekHint"
      tone="gold"
    >
      <template #icon>
        <Trophy aria-hidden="true" />
      </template>
    </MetricTile>
    <MetricTile label="当前热度" :value="heatValue" :hint="heatWeekHint" tone="blue">
      <template #icon>
        <Activity aria-hidden="true" />
      </template>
    </MetricTile>
    <MetricTile label="预测增长" :value="growthValue" :hint="targetWeekHint" tone="red">
      <template #icon>
        <TrendingUp aria-hidden="true" />
      </template>
    </MetricTile>
    <MetricTile label="预测份额" :value="shareValue" :hint="targetWeekHint" tone="neutral">
      <template #icon>
        <PieChart aria-hidden="true" />
      </template>
    </MetricTile>
    <MetricTile
      label="趋势候选"
      :value="eligibleValue"
      :hint="eligibleHint"
      :tone="eligibleTone"
    >
      <template #icon>
        <BadgeCheck v-if="detail?.latest_trend?.is_trend_eligible_t" aria-hidden="true" />
        <CircleDashed v-else aria-hidden="true" />
      </template>
    </MetricTile>
  </section>
</template>

<script setup lang="ts">
import {
  Activity,
  BadgeCheck,
  CircleDashed,
  PieChart,
  TrendingUp,
  Trophy,
} from "lucide-vue-next";
import { computed } from "vue";

import MetricTile from "@/components/ui/MetricTile.vue";
import type { AttributeDetailResponse } from "@/types/api";
import {
  formatNumber,
  formatPercent,
  formatSignedPercent,
  formatWeek,
} from "@/utils/formatters";

const props = defineProps<{
  detail: AttributeDetailResponse | null;
}>();

const latestTrend = computed(() => props.detail?.latest_trend ?? null);
const latestHeat = computed(() => props.detail?.latest_heat ?? null);

const rankValue = computed(() =>
  typeof latestTrend.value?.rank === "number" ? `#${latestTrend.value.rank}` : "--",
);
const heatValue = computed(() => formatNumber(latestHeat.value?.heat, 3));
const growthValue = computed(() =>
  formatSignedPercent(latestTrend.value?.pred_target_growth, 1),
);
const shareValue = computed(() => formatPercent(latestTrend.value?.pred_share_t1, 1));
const eligibleValue = computed(() => {
  if (!latestTrend.value) {
    return "--";
  }
  return latestTrend.value.is_trend_eligible_t ? "是" : "否";
});
const eligibleTone = computed<"green" | "neutral">(() =>
  latestTrend.value?.is_trend_eligible_t ? "green" : "neutral",
);
const eligibleHint = computed(() =>
  latestTrend.value ? "来自趋势样本候选标记" : "无趋势榜记录",
);
const sourceWeekHint = computed(() =>
  latestTrend.value?.source_week ? `${formatWeek(latestTrend.value.source_week)} source` : "--",
);
const targetWeekHint = computed(() =>
  latestTrend.value?.target_week ? `${formatWeek(latestTrend.value.target_week)} target` : "--",
);
const heatWeekHint = computed(() =>
  latestHeat.value?.week_id ? `${formatWeek(latestHeat.value.week_id)} heat` : "--",
);
</script>
